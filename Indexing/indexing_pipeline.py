#!/usr/bin/env python3
"""
RAG Indexing Pipeline using LlamaIndex, OpenAI, and ChromaDB
Handles document chunking, embedding generation, and vector indexing

NOTE: THIS WILL TAKE 16 ROUNDS OF OPENAI API CALLS TO COMPLETE
We have 31,978 nodes and our embedder is working in batches of 2,048 (set my LlamaIndex), so we'll see the progress bar reset roughly 16 times
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Any
from datetime import datetime

from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    Settings
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from chromadb.config import Settings as ChromaSettings
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Clean text by removing navigation artifacts and redundant content"""
    import re
    
    # Remove multiple consecutive newlines
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Remove navigation elements
    navigation_patterns = [
        r'eoPortal\s*\n\s*Satellite Missions\s*\n\s*Other Space Activities\s*\n\s*Search',
        r'Table of contents\s*\n',
        r'Satellite Missions Catalogue\s*\n',
        r'Terms and Conditions\s*\n\s*Cookie Notice\s*\n\s*Privacy Notice\s*\n\s*Leave Feedback\s*\n\s*Contact\s*\n\s*About',
        r'Join our\s*\n\s*Newsletter',
        r'©\s*\d{4}',
        r'Last updated:\s*\n',
        r'Quick facts\s*\n\s*Overview',
        r'\nReferences\n.*?(?=\n\n|\Z)',  # Remove references section
        r'FAQ\n.*?(?=\n\n|\Z)',  # Remove FAQ section at the end
    ]
    
    for pattern in navigation_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Clean up mission ID/source lines
    text = re.sub(r'Mission:\s*[\w-]+\s*\n', '', text)
    text = re.sub(r'Source:\s*https?://[^\s]+\s*\n', '', text)
    
    # Remove [web source no longer available] and similar
    text = re.sub(r'\[web source no longer available\]', '', text)
    text = re.sub(r'URL:\s*\[.*?\]', '', text)
    
    # Clean up excessive whitespace
    text = re.sub(r' +', ' ', text)  # Multiple spaces to single space
    text = re.sub(r'\n +', '\n', text)  # Remove leading spaces on lines
    text = re.sub(r' +\n', '\n', text)  # Remove trailing spaces on lines
    
    # Clean up bullet points and lists
    text = re.sub(r'^\s*[-•]\s*', '- ', text, flags=re.MULTILINE)
    
    # Remove empty parentheses and brackets
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'\[\s*\]', '', text)
    
    # Ensure single newline between sections
    text = re.sub(r'(##[^\n]+)\n\n+', r'\1\n\n', text)
    
    return text.strip()

def markdown_table(rows):
    if not rows:
        return ""
    header = rows[0]
    body = rows[1:]
    out = []
    out.append("| " + " | ".join(str(h) for h in header) + " |")
    out.append("| " + " | ".join(["---"] * len(header)) + " |")
    for r in body:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


class SpaceMissionIndexer:
    """Handles the indexing pipeline for space mission documents"""
    
    def __init__(
        self, 
        data_dir: str,
        chroma_persist_dir: str = "./chroma_db",
        collection_name: str = "space_missions",
        embedding_model: str = "text-embedding-3-large",
        chunk_size: int = 2000,
        chunk_overlap: int = 100
    ):
        """
        Initialize the indexing pipeline
        
        Args:
            data_dir: Directory containing combined mission documents
            chroma_persist_dir: Directory to persist ChromaDB
            collection_name: Name of the ChromaDB collection
            embedding_model: OpenAI embedding model to use
            chunk_size: Size of text chunks in tokens
            chunk_overlap: Overlap between chunks in tokens
        """
        self.data_dir = Path(data_dir)
        self.chroma_persist_dir = chroma_persist_dir
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        
        # Initialize OpenAI settings
        self._setup_openai()
        
        # Initialize ChromaDB
        self._setup_chromadb()
        
        # Configure LlamaIndex settings
        self._configure_llama_index(embedding_model)
        
    def _setup_openai(self):
        """Configure OpenAI API key"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
    def _setup_chromadb(self):
        """Initialize ChromaDB client and collection"""
        chroma_settings = ChromaSettings(
            persist_directory=self.chroma_persist_dir,
            anonymized_telemetry=False
        )
        
        self.chroma_client = chromadb.PersistentClient(
            path=self.chroma_persist_dir,
            settings=chroma_settings
        )
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"ChromaDB collection '{self.collection_name}' ready")
        
    def _configure_llama_index(self, embedding_model: str):
        """Configure LlamaIndex settings"""
        # Set up embedding model
        Settings.embed_model = OpenAIEmbedding(
            model=embedding_model,
            embed_batch_size=50
        )
        
        # Set up LLM (for later use in query pipeline)
        Settings.llm = OpenAI(model="o3", temperature=0)
        
        # Configure chunk settings
        Settings.chunk_size = self.chunk_size
        Settings.chunk_overlap = self.chunk_overlap
        
    def load_documents(self) -> List[Document]:
        """Load all mission documents from the data directory"""
        documents = []
        doc_files = list(self.data_dir.glob("*_combined.json"))
        
        logger.info(f"Found {len(doc_files)} documents to index")
        
        for doc_file in tqdm(doc_files, desc="Loading documents"):
            try:
                with open(doc_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Create document with text and metadata
                text_content = data.get('text', '')
                
                # Clean the text before adding images and tables
                text_content = clean_text(text_content)
                
                # Include image information in the text
                images = data.get('images', []) or []
                if images:
                    text_content += "\n\n## Images:\n"
                    for i, img in enumerate(images):
                        alt_text = img.get('alt', '')
                        caption = img.get('caption', '')
                        
                        if alt_text:
                            text_content += f"\n### {alt_text}\n"
                        else:
                            text_content += f"\n### Image {i+1}\n"
                        
                        if caption and caption.strip():
                            text_content += f"{caption}\n"
                        elif alt_text and not alt_text.startswith('Figure'):
                            # If no caption but alt text contains description
                            text_content += f"{alt_text}\n"
                
                # Include table information in the text with better formatting
                tables = data.get('tables', []) or []
                table_summaries = []
                if tables:
                    text_content += "\n\n## Tables:\n"
                    for i, tbl in enumerate(tables):
                        rows = tbl.get('rows', [])
                        md = markdown_table(rows)
                        caption = data.get("title") or data.get("mission_id") or "Mission table"
                        text_content += f"\n### Table {i+1}: {caption}\n"
                        if md:
                            text_content += md + "\n"
                        n_rows = max(len(rows) - 1, 0) if rows else 0
                        n_cols = len(rows[0]) if rows and len(rows[0]) else 0
                        table_summaries.append({
                            'index': i,
                            'caption': caption,
                            'num_rows': n_rows,
                            'num_cols': n_cols
                        })
                
                # Create LlamaIndex document
                doc = Document(
                    text=text_content,
                    metadata={
                        'mission_id': data['mission_id'],
                        'title': data['title'],
                        'url': data['url'],
                        'has_tables': bool(data.get('tables')),
                        'has_images': bool(data.get('images')),
                        'num_tables': len(data.get('tables', [])),
                        'num_images': len(data.get('images', [])),
                        'table_summaries': json.dumps(table_summaries),
                        'source_file': str(doc_file)
                    }
                )
                documents.append(doc)
                
            except Exception as e:
                logger.error(f"Error loading {doc_file}: {e}")
                
        return documents
    
    def create_nodes(self, documents: List[Document]) -> List[Any]:
        """
        Create nodes (chunks) from documents with special handling for tables
        """
        # Use SentenceSplitter for consistent chunking
        node_parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            include_metadata=True,
            include_prev_next_rel=True,
        )
        
        nodes = []
        for doc in tqdm(documents, desc="Creating chunks"):
            # Get the main document nodes
            doc_nodes = node_parser.get_nodes_from_documents([doc])
            
            # Tag nodes with stable IDs
            for i, node in enumerate(doc_nodes):
                mid = doc.metadata.get('mission_id') or doc.metadata.get('title') or 'mission'
                node.id_ = f"{mid}#ch{i:04d}"
            
            nodes.extend(doc_nodes)
            
        logger.info(f"Created {len(nodes)} chunks from {len(documents)} documents")
        return nodes
    
    
    def create_index(self, nodes: List[Any]) -> VectorStoreIndex:
        """
        Create vector index from nodes using ChromaDB
        """
        # Create ChromaDB vector store
        vector_store = ChromaVectorStore(
            chroma_collection=self.collection
        )
        
        # Create storage context
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )
        
        # Create index
        logger.info("Creating vector index...")
        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            show_progress=True
        )
        
        logger.info("Vector index created successfully")
        return index
    
    def save_index_metadata(self, index: VectorStoreIndex, nodes: List[Any], embedding_model: str):
        """Save metadata about the indexing process"""
        try:
            model_name = Settings.embed_model.model_name
        except:
            model_name = embedding_model
            
        metadata = {
            'indexed_at': datetime.now().isoformat(),
            'num_documents': len(set(node.metadata.get('mission_id') for node in nodes)),
            'num_chunks': len(nodes),
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'embedding_model': model_name,
            'collection_name': self.collection_name,
            'chroma_persist_dir': self.chroma_persist_dir
        }
        
        metadata_path = Path(self.chroma_persist_dir) / 'index_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Index metadata saved to {metadata_path}")
        
    def run_indexing_pipeline(self):
        """Execute the complete indexing pipeline"""
        logger.info("Starting indexing pipeline...")
        
        # Load documents
        documents = self.load_documents()
        if not documents:
            logger.error("No documents found to index")
            return None
            
        # Create nodes (chunks)
        nodes = self.create_nodes(documents)
        
        # Create and persist index
        index = self.create_index(nodes)
        
        # Save metadata
        self.save_index_metadata(index, nodes, self.embedding_model)
        
        logger.info("Indexing pipeline completed successfully!")
        
        # Print summary statistics
        print("\n" + "="*50)
        print("Indexing Summary")
        print("="*50)
        print(f"Documents indexed: {len(documents)}")
        print(f"Total chunks created: {len(nodes)}")
        print(f"Average chunks per document: {len(nodes) / len(documents):.1f}")
        print(f"Collection name: {self.collection_name}")
        print(f"Persist directory: {self.chroma_persist_dir}")
        
        return index


def main():
    """Main function to run the indexing pipeline"""
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it using: export OPENAI_API_KEY='your-api-key'")
        return
    
    # Determine if we're running from the Indexing directory or parent directory
    current_dir = Path.cwd()
    if current_dir.name == "Indexing":
        # Running from within Indexing directory
        data_dir = "../rag_ready_data/combined_documents"
        chroma_persist_dir = "../chroma_db"
    else:
        # Running from parent directory (SpaceMissionDesignAssistant)
        data_dir = "./rag_ready_data/combined_documents"
        chroma_persist_dir = "./chroma_db"
    
    # Check if data directory exists
    if not Path(data_dir).exists():
        print(f"Error: Data directory {data_dir} not found")
        print("Please run Indexing/prepare_rag_data.py first to prepare the data")
        return
    
    # Initialize indexer
    indexer = SpaceMissionIndexer(
        data_dir=data_dir,
        chroma_persist_dir=chroma_persist_dir,
        collection_name="space_missions",
        embedding_model="text-embedding-3-large",
        chunk_size=2000,
        chunk_overlap=100
    )
    
    # Run indexing
    index = indexer.run_indexing_pipeline()
    
    if index:
        print("\n✓ Indexing complete! The vector database is ready for queries.")
        print(f"✓ ChromaDB persisted at: {chroma_persist_dir}")
        print("\nNext step: Run query_pipeline.py to query the indexed data")


if __name__ == "__main__":
    main()