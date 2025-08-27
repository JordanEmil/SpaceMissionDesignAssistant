#!/usr/bin/env python3
"""
RAG Evaluation Framework for Space Mission Knowledge Base
Implements evaluation metrics from literature review including retrieval and generation metrics
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, asdict
import asyncio

from llama_index.core import QueryBundle
from llama_index.core.evaluation import (
    RelevancyEvaluator,
    FaithfulnessEvaluator,
    RetrieverEvaluator,
    EmbeddingRetrievalEvaluator,
    SemanticSimilarityEvaluator,
    BatchEvalRunner
)
from llama_index.core.evaluation.retrieval.metrics import (
    MRR,
    HitRate,
    MAP
)
from llama_index.llms.openai import OpenAI
from tqdm import tqdm
import pandas as pd

from query_pipeline import SpaceMissionQueryEngine
from Indexing.indexing_pipeline import SpaceMissionIndexer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    query: str
    expected_answer: Optional[str]
    generated_answer: str
    retrieved_contexts: List[str]
    relevant_contexts: Optional[List[str]]
    
    # Retrieval metrics
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    map_score: float
    mrr_score: float
    hit_rate: float
    
    # Generation metrics
    exact_match: float
    f1_score: float
    semantic_similarity: float
    
    # LLM-based metrics
    relevancy_score: Optional[float]
    faithfulness_score: Optional[float]
    
    # Metadata
    response_time: float
    timestamp: str


class RAGEvaluator:
    """Comprehensive evaluation framework for RAG system"""
    
    def __init__(
        self,
        query_engine: SpaceMissionQueryEngine,
        llm_model: str = "gpt-4o",
        evaluation_data_path: Optional[str] = None
    ):
        """
        Initialize the RAG evaluator
        
        Args:
            query_engine: The query engine to evaluate
            llm_model: LLM model for evaluation
            evaluation_data_path: Path to evaluation dataset
        """
        self.query_engine = query_engine
        self.llm = OpenAI(model=llm_model, temperature=0)
        self.evaluation_data_path = evaluation_data_path
        
        # Initialize LlamaIndex evaluators
        self.relevancy_evaluator = RelevancyEvaluator(llm=self.llm)
        self.faithfulness_evaluator = FaithfulnessEvaluator(llm=self.llm)
        self.semantic_evaluator = SemanticSimilarityEvaluator()
        
        # Results storage
        self.results = []
        
    def load_evaluation_dataset(self) -> List[Dict[str, Any]]:
        """Load or generate evaluation dataset"""
        if self.evaluation_data_path and Path(self.evaluation_data_path).exists():
            with open(self.evaluation_data_path, 'r') as f:
                return json.load(f)
        else:
            # Generate sample evaluation dataset
            return self.generate_evaluation_dataset()
    
    def generate_evaluation_dataset(self) -> List[Dict[str, Any]]:
        """Generate synthetic evaluation dataset based on indexed content"""
        eval_questions = [
            {
                "query": "What are the typical orbit parameters for SAR imaging satellites?",
                "expected_contexts": ["orbit", "SAR", "altitude", "inclination"],
                "expected_answer_keywords": ["sun-synchronous", "polar", "altitude", "repeat cycle"]
            },
            {
                "query": "What power systems are commonly used in CubeSats?",
                "expected_contexts": ["power", "CubeSat", "solar", "battery"],
                "expected_answer_keywords": ["solar panels", "battery", "EPS", "power budget"]
            },
            {
                "query": "Which missions have demonstrated optical imaging from LEO?",
                "expected_contexts": ["optical", "imaging", "LEO", "Earth observation"],
                "expected_answer_keywords": ["resolution", "spectral bands", "swath width"]
            },
            {
                "query": "What are the main components of a satellite communication system?",
                "expected_contexts": ["communication", "antenna", "transponder", "ground station"],
                "expected_answer_keywords": ["antenna", "transceiver", "modulation", "frequency"]
            },
            {
                "query": "How do formation flying missions maintain relative positioning?",
                "expected_contexts": ["formation flying", "relative position", "control", "GPS"],
                "expected_answer_keywords": ["GPS", "inter-satellite link", "propulsion", "control algorithm"]
            }
        ]
        
        # Add more domain-specific questions
        additional_questions = [
            {
                "query": "What are typical data rates for Earth observation satellite downlinks?",
                "expected_contexts": ["data rate", "downlink", "communication", "ground station"],
                "expected_answer_keywords": ["Mbps", "X-band", "Ka-band", "ground station"]
            },
            {
                "query": "Which propulsion systems are used for small satellite orbit maintenance?",
                "expected_contexts": ["propulsion", "small satellite", "orbit maintenance", "thruster"],
                "expected_answer_keywords": ["electric propulsion", "cold gas", "ion thruster", "delta-v"]
            },
            {
                "query": "What are the advantages of constellation missions for Earth observation?",
                "expected_contexts": ["constellation", "coverage", "revisit time", "Earth observation"],
                "expected_answer_keywords": ["global coverage", "revisit time", "redundancy", "temporal resolution"]
            },
            {
                "query": "How do star trackers work for satellite attitude determination?",
                "expected_contexts": ["star tracker", "attitude", "sensor", "navigation"],
                "expected_answer_keywords": ["star catalog", "CCD", "quaternion", "accuracy"]
            },
            {
                "query": "What thermal control methods are used in spacecraft design?",
                "expected_contexts": ["thermal", "temperature", "radiator", "insulation"],
                "expected_answer_keywords": ["MLI", "radiator", "heater", "thermal coating"]
            }
        ]
        
        eval_questions.extend(additional_questions)
        
        # Generate ground truth answers (in real scenario, these would be manually created)
        for question in eval_questions:
            question["ground_truth_answer"] = self._generate_ground_truth(question["query"])
            
        return eval_questions
    
    def _generate_ground_truth(self, query: str) -> str:
        """Generate ground truth answer using the query engine"""
        # In practice, these would be manually curated
        # Here we use the system itself for demonstration
        result = self.query_engine.query(query, verbose=False)
        return result['response']
    
    def calculate_exact_match(self, predicted: str, ground_truth: str) -> float:
        """Calculate exact match score"""
        # Normalize strings
        pred_normalized = predicted.strip().lower()
        truth_normalized = ground_truth.strip().lower()
        
        return 1.0 if pred_normalized == truth_normalized else 0.0
    
    def calculate_f1_score(self, predicted: str, ground_truth: str) -> float:
        """Calculate token-level F1 score"""
        # Tokenize
        pred_tokens = set(predicted.lower().split())
        truth_tokens = set(ground_truth.lower().split())
        
        if not pred_tokens or not truth_tokens:
            return 0.0
        
        # Calculate precision and recall
        common_tokens = pred_tokens.intersection(truth_tokens)
        precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
        recall = len(common_tokens) / len(truth_tokens) if truth_tokens else 0
        
        # Calculate F1
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def calculate_retrieval_metrics(
        self,
        retrieved_docs: List[Dict[str, Any]],
        relevant_docs: List[str],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, Any]:
        """Calculate retrieval metrics: Precision@k, Recall@k, MAP, MRR"""
        metrics = {
            'precision_at_k': {},
            'recall_at_k': {},
            'map': 0.0,
            'mrr': 0.0,
            'hit_rate': 0.0
        }
        
        # Convert retrieved docs to comparable format
        retrieved_ids = [doc['metadata'].get('mission_id', '') for doc in retrieved_docs]
        
        # Calculate metrics for different k values
        for k in k_values:
            top_k_retrieved = retrieved_ids[:k]
            
            # Precision@k
            relevant_in_top_k = sum(1 for doc_id in top_k_retrieved if doc_id in relevant_docs)
            metrics['precision_at_k'][k] = relevant_in_top_k / k if k > 0 else 0
            
            # Recall@k
            metrics['recall_at_k'][k] = relevant_in_top_k / len(relevant_docs) if relevant_docs else 0
        
        # Mean Average Precision (MAP)
        average_precisions = []
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_docs:
                # Precision at this position
                precision_at_i = sum(1 for d in retrieved_ids[:i+1] if d in relevant_docs) / (i + 1)
                average_precisions.append(precision_at_i)
        
        metrics['map'] = np.mean(average_precisions) if average_precisions else 0.0
        
        # Mean Reciprocal Rank (MRR)
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_docs:
                metrics['mrr'] = 1.0 / (i + 1)
                break
        
        # Hit Rate
        metrics['hit_rate'] = 1.0 if any(doc_id in relevant_docs for doc_id in retrieved_ids) else 0.0
        
        return metrics
    
    async def evaluate_single_query(
        self,
        eval_item: Dict[str, Any],
        use_llm_evaluation: bool = True
    ) -> EvaluationResult:
        """Evaluate a single query comprehensively"""
        query = eval_item["query"]
        
        # Execute query
        import time
        start_time = time.time()
        result = self.query_engine.query(
            query,
            return_sources=True,
            verbose=False
        )
        response_time = time.time() - start_time
        
        # Extract information
        generated_answer = result['response']
        retrieved_contexts = result.get('sources', [])
        
        # Calculate retrieval metrics
        relevant_docs = eval_item.get('relevant_doc_ids', [])
        retrieval_metrics = self.calculate_retrieval_metrics(
            retrieved_contexts,
            relevant_docs
        )
        
        # Calculate generation metrics
        ground_truth = eval_item.get('ground_truth_answer', '')
        exact_match = self.calculate_exact_match(generated_answer, ground_truth)
        f1_score = self.calculate_f1_score(generated_answer, ground_truth)
        
        # Calculate semantic similarity
        semantic_sim = 0.0
        if ground_truth:
            # Use embedding similarity
            semantic_sim = await self.semantic_evaluator.aevaluate(
                response=generated_answer,
                reference=ground_truth
            )
            semantic_sim = semantic_sim.score if hasattr(semantic_sim, 'score') else semantic_sim
        
        # LLM-based evaluation (optional due to cost)
        relevancy_score = None
        faithfulness_score = None
        
        if use_llm_evaluation and retrieved_contexts:
            # Evaluate relevancy
            relevancy_result = await self.relevancy_evaluator.aevaluate(
                query=query,
                response=generated_answer
            )
            relevancy_score = relevancy_result.score if hasattr(relevancy_result, 'score') else None
            
            # Evaluate faithfulness
            context_str = "\n".join([ctx['text'] for ctx in retrieved_contexts[:3]])
            faithfulness_result = await self.faithfulness_evaluator.aevaluate(
                query=query,
                response=generated_answer,
                contexts=[context_str]
            )
            faithfulness_score = faithfulness_result.score if hasattr(faithfulness_result, 'score') else None
        
        # Create evaluation result
        eval_result = EvaluationResult(
            query=query,
            expected_answer=ground_truth,
            generated_answer=generated_answer,
            retrieved_contexts=[ctx['text'] for ctx in retrieved_contexts],
            relevant_contexts=relevant_docs,
            precision_at_k=retrieval_metrics['precision_at_k'],
            recall_at_k=retrieval_metrics['recall_at_k'],
            map_score=retrieval_metrics['map'],
            mrr_score=retrieval_metrics['mrr'],
            hit_rate=retrieval_metrics['hit_rate'],
            exact_match=exact_match,
            f1_score=f1_score,
            semantic_similarity=float(semantic_sim),
            relevancy_score=float(relevancy_score) if relevancy_score else None,
            faithfulness_score=float(faithfulness_score) if faithfulness_score else None,
            response_time=response_time,
            timestamp=datetime.now().isoformat()
        )
        
        return eval_result
    
    def run_evaluation(
        self,
        num_queries: Optional[int] = None,
        use_llm_evaluation: bool = True,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """Run complete evaluation on the dataset"""
        # Load evaluation dataset
        eval_dataset = self.load_evaluation_dataset()
        
        if num_queries:
            eval_dataset = eval_dataset[:num_queries]
        
        logger.info(f"Running evaluation on {len(eval_dataset)} queries")
        
        # Run evaluations
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        tasks = []
        for eval_item in tqdm(eval_dataset, desc="Evaluating queries"):
            task = self.evaluate_single_query(eval_item, use_llm_evaluation)
            tasks.append(task)
        
        # Execute all evaluations
        self.results = loop.run_until_complete(asyncio.gather(*tasks))
        loop.close()
        
        # Calculate aggregate metrics
        aggregate_metrics = self.calculate_aggregate_metrics()
        
        # Save results if requested
        if save_results:
            self.save_evaluation_results(aggregate_metrics)
        
        return aggregate_metrics
    
    def calculate_aggregate_metrics(self) -> Dict[str, Any]:
        """Calculate aggregate metrics across all queries"""
        if not self.results:
            return {}
        
        # Initialize aggregators
        metrics = defaultdict(list)
        precision_at_k = defaultdict(list)
        recall_at_k = defaultdict(list)
        
        # Collect metrics from all results
        for result in self.results:
            metrics['exact_match'].append(result.exact_match)
            metrics['f1_score'].append(result.f1_score)
            metrics['semantic_similarity'].append(result.semantic_similarity)
            metrics['map_score'].append(result.map_score)
            metrics['mrr_score'].append(result.mrr_score)
            metrics['hit_rate'].append(result.hit_rate)
            metrics['response_time'].append(result.response_time)
            
            if result.relevancy_score is not None:
                metrics['relevancy_score'].append(result.relevancy_score)
            if result.faithfulness_score is not None:
                metrics['faithfulness_score'].append(result.faithfulness_score)
            
            # Collect precision/recall at different k values
            for k, p in result.precision_at_k.items():
                precision_at_k[k].append(p)
            for k, r in result.recall_at_k.items():
                recall_at_k[k].append(r)
        
        # Calculate averages
        aggregate = {
            'num_queries': len(self.results),
            'retrieval_metrics': {
                'mean_map': np.mean(metrics['map_score']),
                'mean_mrr': np.mean(metrics['mrr_score']),
                'mean_hit_rate': np.mean(metrics['hit_rate']),
                'precision_at_k': {k: np.mean(v) for k, v in precision_at_k.items()},
                'recall_at_k': {k: np.mean(v) for k, v in recall_at_k.items()}
            },
            'generation_metrics': {
                'mean_exact_match': np.mean(metrics['exact_match']),
                'mean_f1_score': np.mean(metrics['f1_score']),
                'mean_semantic_similarity': np.mean(metrics['semantic_similarity'])
            },
            'llm_evaluation_metrics': {},
            'performance_metrics': {
                'mean_response_time': np.mean(metrics['response_time']),
                'p95_response_time': np.percentile(metrics['response_time'], 95),
                'p99_response_time': np.percentile(metrics['response_time'], 99)
            }
        }
        
        # Add LLM metrics if available
        if metrics['relevancy_score']:
            aggregate['llm_evaluation_metrics']['mean_relevancy'] = np.mean(metrics['relevancy_score'])
        if metrics['faithfulness_score']:
            aggregate['llm_evaluation_metrics']['mean_faithfulness'] = np.mean(metrics['faithfulness_score'])
        
        return aggregate
    
    def save_evaluation_results(self, aggregate_metrics: Dict[str, Any]):
        """Save evaluation results to files"""
        output_dir = Path("evaluation_results")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save aggregate metrics
        with open(output_dir / f"aggregate_metrics_{timestamp}.json", 'w') as f:
            json.dump(aggregate_metrics, f, indent=2)
        
        # Save detailed results
        detailed_results = []
        for result in self.results:
            detailed_results.append(asdict(result))
        
        with open(output_dir / f"detailed_results_{timestamp}.json", 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        # Create results DataFrame for analysis
        results_df = pd.DataFrame([
            {
                'query': r.query,
                'exact_match': r.exact_match,
                'f1_score': r.f1_score,
                'semantic_similarity': r.semantic_similarity,
                'map_score': r.map_score,
                'mrr_score': r.mrr_score,
                'response_time': r.response_time,
                'relevancy_score': r.relevancy_score,
                'faithfulness_score': r.faithfulness_score
            }
            for r in self.results
        ])
        
        results_df.to_csv(output_dir / f"evaluation_results_{timestamp}.csv", index=False)
        
        logger.info(f"Evaluation results saved to {output_dir}")
    
    def print_evaluation_summary(self, metrics: Dict[str, Any]):
        """Print a formatted summary of evaluation results"""
        print("\n" + "="*60)
        print("RAG System Evaluation Summary")
        print("="*60)
        
        print(f"\nTotal queries evaluated: {metrics['num_queries']}")
        
        print("\n--- Retrieval Metrics ---")
        print(f"Mean Average Precision (MAP): {metrics['retrieval_metrics']['mean_map']:.3f}")
        print(f"Mean Reciprocal Rank (MRR): {metrics['retrieval_metrics']['mean_mrr']:.3f}")
        print(f"Hit Rate: {metrics['retrieval_metrics']['mean_hit_rate']:.3f}")
        
        print("\nPrecision@k:")
        for k, p in sorted(metrics['retrieval_metrics']['precision_at_k'].items()):
            print(f"  P@{k}: {p:.3f}")
        
        print("\nRecall@k:")
        for k, r in sorted(metrics['retrieval_metrics']['recall_at_k'].items()):
            print(f"  R@{k}: {r:.3f}")
        
        print("\n--- Generation Metrics ---")
        print(f"Exact Match: {metrics['generation_metrics']['mean_exact_match']:.3f}")
        print(f"F1 Score: {metrics['generation_metrics']['mean_f1_score']:.3f}")
        print(f"Semantic Similarity: {metrics['generation_metrics']['mean_semantic_similarity']:.3f}")
        
        if metrics['llm_evaluation_metrics']:
            print("\n--- LLM Evaluation Metrics ---")
            if 'mean_relevancy' in metrics['llm_evaluation_metrics']:
                print(f"Relevancy: {metrics['llm_evaluation_metrics']['mean_relevancy']:.3f}")
            if 'mean_faithfulness' in metrics['llm_evaluation_metrics']:
                print(f"Faithfulness: {metrics['llm_evaluation_metrics']['mean_faithfulness']:.3f}")
        
        print("\n--- Performance Metrics ---")
        print(f"Mean Response Time: {metrics['performance_metrics']['mean_response_time']:.2f}s")
        print(f"P95 Response Time: {metrics['performance_metrics']['p95_response_time']:.2f}s")
        print(f"P99 Response Time: {metrics['performance_metrics']['p99_response_time']:.2f}s")


def main():
    """Main function to run evaluation"""
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    # Initialize query engine
    print("Initializing query engine...")
    query_engine = SpaceMissionQueryEngine(
        chroma_persist_dir="./chroma_db",
        top_k=5,
        similarity_threshold=0.7
    )
    
    # Initialize evaluator
    print("\nInitializing evaluation framework...")
    evaluator = RAGEvaluator(
        query_engine=query_engine,
        llm_model="gpt-4o"
    )
    
    # Run evaluation
    print("\nRunning evaluation...")
    print("Note: LLM-based evaluation is disabled by default to save costs.")
    print("Set use_llm_evaluation=True to enable relevancy and faithfulness scoring.\n")
    
    metrics = evaluator.run_evaluation(
        num_queries=10,  # Limit for demonstration
        use_llm_evaluation=False,  # Disable to save API costs
        save_results=True
    )
    
    # Print summary
    evaluator.print_evaluation_summary(metrics)
    
    print("\n✓ Evaluation complete!")
    print("✓ Results saved to: evaluation_results/")


if __name__ == "__main__":
    main()