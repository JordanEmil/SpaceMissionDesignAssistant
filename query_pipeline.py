#!/usr/bin/env python3
"""
RAG Query Pipeline using LlamaIndex
Handles query processing, retrieval, and answer generation
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import time

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import (
    get_response_synthesizer,
    ResponseMode
)
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from chromadb.config import Settings as ChromaSettings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpaceMissionQueryEngine:
    """Handles queries against the indexed space mission database"""
    
    def __init__(
        self,
        chroma_persist_dir: str = "./chroma_db",
        collection_name: str = "space_missions",
        embedding_model: str = "text-embedding-3-large",
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        top_k: int = 5,
        similarity_threshold: float = 0.35
    ):
        """
        Initialize the query engine
        
        Args:
            chroma_persist_dir: Directory where ChromaDB is persisted
            collection_name: Name of the ChromaDB collection
            embedding_model: OpenAI embedding model
            llm_model: OpenAI LLM model for generation
            temperature: Temperature for LLM generation
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score for retrieved documents
        """
        self.chroma_persist_dir = chroma_persist_dir
        self.collection_name = collection_name
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.temperature = temperature
        
        # Initialize components
        self._setup_openai()
        self._configure_llama_index(embedding_model, llm_model, temperature)
        self._setup_chromadb()
        self._load_index()
        
    def _setup_openai(self):
        """Configure OpenAI API key"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
            
    def _configure_llama_index(self, embedding_model: str, llm_model: str, temperature: float):
        """Configure LlamaIndex settings"""
        # Set up embedding model
        Settings.embed_model = OpenAIEmbedding(
            model=embedding_model,
            embed_batch_size=100
        )
        
        # Set up LLM
        Settings.llm = OpenAI(
            model=llm_model,
            temperature=temperature,
            max_tokens=2048
        )
        
    def _setup_chromadb(self):
        """Initialize ChromaDB client"""
        chroma_settings = ChromaSettings(
            persist_directory=self.chroma_persist_dir,
            anonymized_telemetry=False
        )
        
        self.chroma_client = chromadb.PersistentClient(
            path=self.chroma_persist_dir,
            settings=chroma_settings
        )
        
        # Get collection
        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            logger.info(f"Loaded ChromaDB collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Failed to load collection: {e}")
            raise
            
    def _load_index(self):
        """Load the vector index from ChromaDB"""
        # Create vector store
        vector_store = ChromaVectorStore(
            chroma_collection=self.collection
        )
        
        # Create storage context
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )
        
        # Create index from vector store
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context
        )
        
        logger.info("Vector index loaded successfully")
        
    def create_query_engine(
        self,
        response_mode: ResponseMode = ResponseMode.COMPACT,
        metadata_filters: Optional[Dict[str, Any]] = None,
        streaming: bool = False
    ):
        """
        Create a configured query engine
        
        Args:
            response_mode: How to synthesize the response
            metadata_filters: Metadata filters to apply
            streaming: Whether to stream the response
        """
        # Apply metadata filters if provided
        filters = None
        if metadata_filters:
            filters = MetadataFilters(
                filters=[ExactMatchFilter(key=k, value=v) for k, v in metadata_filters.items()]
            )
        
        # Create retriever
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.top_k,
            filters=filters
        )
        
        # Create response synthesizer
        response_synthesizer = get_response_synthesizer(
            response_mode=response_mode,
            streaming=streaming,
            use_async=False
        )
        
        # Create post-processors
        postprocessors = [
            SimilarityPostprocessor(similarity_cutoff=self.similarity_threshold)
        ]
        
        # Create query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=postprocessors
        )
        
        return query_engine
    
    def query(
        self,
        query_text: str,
        response_mode: ResponseMode = ResponseMode.COMPACT,
        metadata_filters: Optional[Dict[str, Any]] = None,
        return_sources: bool = True,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a query against the index
        
        Args:
            query_text: The query string
            response_mode: How to synthesize the response
            metadata_filters: Filters to apply to the retrieval
            return_sources: Whether to return source documents
            verbose: Whether to print verbose output
            
        Returns:
            Dictionary containing response and metadata
        """
        start_time = time.time()
        
        # Create query engine with metadata filters
        query_engine = self.create_query_engine(
            response_mode=response_mode, 
            metadata_filters=metadata_filters
        )
        
        # Execute query
        if verbose:
            print(f"\nQuerying: {query_text}")
            print(f"Retrieving top {self.top_k} documents...")
        
        response = query_engine.query(query_text)
        
        # Extract results
        result = {
            'query': query_text,
            'response': str(response),
            'metadata': {
                'response_time': time.time() - start_time,
                'model': Settings.llm.model,
                'temperature': self.temperature,
                'top_k': self.top_k,
                'response_mode': response_mode.value,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # Add source documents if requested
        if return_sources and hasattr(response, 'source_nodes'):
            sources = []
            for node in response.source_nodes:
                source = {
                    'text': node.text[:500] + "..." if len(node.text) > 500 else node.text,
                    'metadata': node.metadata,
                    'score': node.score
                }
                sources.append(source)
            result['sources'] = sources
            
            if verbose:
                print(f"\nRetrieved {len(sources)} source documents")
                for i, source in enumerate(sources):
                    print(f"\n--- Source {i+1} (score: {source['score']:.3f}) ---")
                    print(f"Mission: {source['metadata'].get('title', 'Unknown')}")
                    print(f"Text preview: {source['text'][:200]}...")
        
        if verbose:
            print(f"\nResponse generated in {result['metadata']['response_time']:.2f} seconds")
        
        return result
    
    def batch_query(
        self,
        queries: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple queries in batch
        
        Args:
            queries: List of query strings
            **kwargs: Additional arguments passed to query()
            
        Returns:
            List of query results
        """
        results = []
        for query in queries:
            result = self.query(query, **kwargs)
            results.append(result)
        return results
    
    def get_similar_missions(
        self,
        mission_id: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find missions similar to a given mission
        
        Args:
            mission_id: ID of the mission to find similar ones to
            top_k: Number of similar missions to return
        """
        # This would require implementing a custom retrieval based on mission embedding
        # For now, return a placeholder
        logger.warning("Similar mission search not yet implemented")
        return []
    
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get statistics about the query engine"""
        collection_stats = self.collection.count()
        
        return {
            'collection_name': self.collection_name,
            'total_chunks': collection_stats,
            'embedding_model': Settings.embed_model.model_name,
            'llm_model': Settings.llm.model,
            'temperature': self.temperature,
            'top_k': self.top_k,
            'similarity_threshold': self.similarity_threshold
        }


def demo_queries():
    """Run demonstration queries"""
    # Sample queries for space mission design
    demo_questions = [
        # "What orbit regimes have been used for SAR imaging satellites? Include specific missions and sources.",
        "What are typical power requirements for Earth observation CubeSats? Include specific missions and sources.",
        "Which missions have used optical imaging payloads in sun-synchronous orbits? Include specific missions and sources.",
        "What are common failure modes in small satellite missions? Include specific missions and sources.",
        "Compare the antenna designs used in different SAR missions? Include specific missions and sources."
    ]
    
    return demo_questions


def main():
    """Main function to demonstrate the query pipeline"""
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    # Check if index exists
    chroma_persist_dir = "./chroma_db"
    if not Path(chroma_persist_dir).exists():
        print(f"Error: ChromaDB directory {chroma_persist_dir} not found")
        print("Please run Indexing/indexing_pipeline.py first to create the index")
        return
    
    # Initialize query engine
    print("Initializing query engine...")
    query_engine = SpaceMissionQueryEngine(
        chroma_persist_dir=chroma_persist_dir,
        top_k=5,
        similarity_threshold=0.35,
        temperature=0.0
    )
    
    # Print engine stats
    stats = query_engine.get_engine_stats()
    print("\n" + "="*50)
    print("Query Engine Configuration")
    print("="*50)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Run demo queries
    print("\n" + "="*50)
    print("Running Demo Queries")
    print("="*50)
    
    demo_questions = demo_queries()
    
    # Run general queries
    for i, question in enumerate(demo_questions[:3]):  # First 3 general queries
        print(f"\n{'='*50}")
        print(f"Query {i+1}: {question}")
        print('='*50)
        
        result = query_engine.query(
            question,
            response_mode=ResponseMode.COMPACT,
            return_sources=True,
            verbose=True
        )
        
        print(f"\nAnswer:\n{result['response']}")
        
        # Save result
        output_dir = Path("query_results")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / f"query_{i+1}_result.json", 'w') as f:
            json.dump(result, f, indent=2)
    
    
    print("\n✓ Query pipeline demonstration complete!")
    print(f"✓ Results saved to: query_results/")


if __name__ == "__main__":
    main()