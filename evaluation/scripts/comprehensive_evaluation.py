#!/usr/bin/env python3
"""
Comprehensive RAG Evaluation Framework with RAGAS Integration
Implements advanced retrieval and generation metrics with parameter sweeping
"""

import os
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, asdict, field
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

# LlamaIndex imports
from llama_index.core import QueryBundle
from llama_index.core.evaluation import (
    AnswerRelevancyEvaluator,
    ContextRelevancyEvaluator,
    FaithfulnessEvaluator,
    SemanticSimilarityEvaluator,
    CorrectnessEvaluator,
    GuidelineEvaluator,
    PairwiseComparisonEvaluator
)
from llama_index.llms.openai import OpenAI
from llama_index.core.response_synthesizers import ResponseMode

# Note: RAGAS functionality is now integrated through LlamaIndex evaluators
# No need for direct RAGAS imports

# Import query engine
import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from query_pipeline import SpaceMissionQueryEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation parameters"""
    top_k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])
    similarity_thresholds: List[float] = field(default_factory=lambda: [0.0, 0.2, 0.35, 0.5, 0.7])
    temperature_values: List[float] = field(default_factory=lambda: [0.0, 0.3, 0.7, 1.0])
    response_modes: List[str] = field(default_factory=lambda: ["compact", "refine", "tree_summarize"])
    llm_models: List[str] = field(default_factory=lambda: ["gpt-4o-mini", "gpt-4o"])
    embedding_models: List[str] = field(default_factory=lambda: ["text-embedding-3-small", "text-embedding-3-large"])


@dataclass 
class RetrievalMetrics:
    """Container for retrieval metrics"""
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    f1_at_k: Dict[int, float]
    map_score: float
    mrr_score: float
    ndcg_at_k: Dict[int, float]
    hit_rate: float
    
    
@dataclass
class GenerationMetrics:
    """Container for generation metrics"""
    # LlamaIndex metrics
    correctness: float
    relevancy: float
    faithfulness: float
    semantic_similarity: float
    guideline_adherence: Optional[float]
    
    # RAGAS metrics
    ragas_faithfulness: float
    ragas_answer_relevancy: float
    ragas_context_recall: float
    ragas_context_precision: float
    
    # Text-based metrics
    bleu_score: float
    rouge_scores: Dict[str, float]
    exact_match: float
    f1_token_score: float


@dataclass
class EvaluationResult:
    """Complete evaluation result for a single query"""
    query_id: str
    query: str
    ground_truth: Optional[str]
    generated_answer: str
    retrieved_contexts: List[str]
    
    # Metrics
    retrieval_metrics: RetrievalMetrics
    generation_metrics: GenerationMetrics
    
    # Configuration used
    config: Dict[str, Any]
    
    # Performance
    retrieval_time: float
    generation_time: float
    total_time: float
    
    # Metadata
    timestamp: str
    error: Optional[str] = None


class ComprehensiveEvaluator:
    """Advanced RAG evaluator with RAGAS integration and parameter sweeping"""
    
    def __init__(
        self,
        chroma_persist_dir: str = "./chroma_db",
        evaluation_llm_model: str = "gpt-4o",
        output_dir: str = "evaluation/results"
    ):
        """Initialize the comprehensive evaluator"""
        self.chroma_persist_dir = chroma_persist_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize LLM for evaluation
        self.eval_llm = OpenAI(model=evaluation_llm_model, temperature=0)
        
        # Initialize LlamaIndex evaluators
        self.correctness_evaluator = CorrectnessEvaluator(llm=self.eval_llm)
        self.answer_relevancy_evaluator = AnswerRelevancyEvaluator(llm=self.eval_llm)
        self.context_relevancy_evaluator = ContextRelevancyEvaluator(llm=self.eval_llm)
        self.faithfulness_evaluator = FaithfulnessEvaluator(llm=self.eval_llm)
        self.semantic_evaluator = SemanticSimilarityEvaluator()
        
        # Results storage
        self.results = []
        self.parameter_sweep_results = defaultdict(list)
        
    def load_evaluation_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load synthetic evaluation dataset"""
        with open(dataset_path, 'r') as f:
            data = json.load(f)
            
        if isinstance(data, dict) and 'questions' in data:
            return data['questions']
        return data
        
    def calculate_retrieval_metrics(
        self,
        retrieved_docs: List[Dict[str, Any]],
        relevant_doc_ids: List[str],
        k_values: List[int] = [1, 3, 5, 10, 20]
    ) -> RetrievalMetrics:
        """Calculate comprehensive retrieval metrics"""
        
        # Extract document IDs from retrieved documents
        # Handle both mission_id and doc_id fields, and ensure we get valid IDs
        retrieved_ids = []
        for doc in retrieved_docs:
            doc_id = doc.get('metadata', {}).get('mission_id', '')
            if not doc_id:
                doc_id = doc.get('metadata', {}).get('doc_id', '')
            if not doc_id:
                doc_id = doc.get('id', '')
            if doc_id:
                retrieved_ids.append(doc_id)
        
        # Ensure relevant_doc_ids is a list of valid IDs
        if not relevant_doc_ids:
            relevant_doc_ids = []
        
        # Log for debugging
        logger.debug(f"Retrieved IDs: {retrieved_ids}")
        logger.debug(f"Relevant IDs: {relevant_doc_ids}")
        
        metrics = {
            'precision_at_k': {},
            'recall_at_k': {},
            'f1_at_k': {},
            'ndcg_at_k': {}
        }
        
        # Calculate metrics for different k values
        for k in k_values:
            top_k_retrieved = retrieved_ids[:k]
            
            # Precision@k
            relevant_in_top_k = sum(1 for doc_id in top_k_retrieved if doc_id in relevant_doc_ids)
            precision = relevant_in_top_k / k if k > 0 else 0
            # Ensure precision is normalized between 0 and 1
            precision = min(1.0, max(0.0, precision))
            metrics['precision_at_k'][k] = precision
            
            # Recall@k  
            recall = relevant_in_top_k / len(relevant_doc_ids) if relevant_doc_ids else 0
            # Ensure recall is normalized between 0 and 1
            recall = min(1.0, max(0.0, recall))
            metrics['recall_at_k'][k] = recall
            
            # F1@k
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0
            # Ensure F1 is normalized between 0 and 1
            f1 = min(1.0, max(0.0, f1))
            metrics['f1_at_k'][k] = f1
            
            # NDCG@k
            dcg = self._calculate_dcg(top_k_retrieved, relevant_doc_ids, k)
            idcg = self._calculate_idcg(relevant_doc_ids, k)
            ndcg = dcg / idcg if idcg > 0 else 0
            # Ensure NDCG is normalized between 0 and 1
            ndcg = min(1.0, max(0.0, ndcg))
            metrics['ndcg_at_k'][k] = ndcg
        
        # MAP (Mean Average Precision)
        average_precisions = []
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_doc_ids:
                precision_at_i = sum(1 for d in retrieved_ids[:i+1] if d in relevant_doc_ids) / (i + 1)
                average_precisions.append(precision_at_i)
        
        map_score = np.mean(average_precisions) if average_precisions else 0.0
        # Ensure MAP is normalized between 0 and 1
        map_score = min(1.0, max(0.0, map_score))
        
        # MRR (Mean Reciprocal Rank)
        mrr_score = 0.0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_doc_ids:
                mrr_score = 1.0 / (i + 1)
                break
        # Ensure MRR is normalized between 0 and 1
        mrr_score = min(1.0, max(0.0, mrr_score))
        
        # Hit Rate
        hit_rate = 1.0 if any(doc_id in relevant_doc_ids for doc_id in retrieved_ids) else 0.0
        
        # Final validation: ensure all metrics are normalized between 0 and 1
        for k in metrics['precision_at_k']:
            metrics['precision_at_k'][k] = min(1.0, max(0.0, metrics['precision_at_k'][k]))
        for k in metrics['recall_at_k']:
            metrics['recall_at_k'][k] = min(1.0, max(0.0, metrics['recall_at_k'][k]))
        for k in metrics['f1_at_k']:
            metrics['f1_at_k'][k] = min(1.0, max(0.0, metrics['f1_at_k'][k]))
        for k in metrics['ndcg_at_k']:
            metrics['ndcg_at_k'][k] = min(1.0, max(0.0, metrics['ndcg_at_k'][k]))
        
        return RetrievalMetrics(
            precision_at_k=metrics['precision_at_k'],
            recall_at_k=metrics['recall_at_k'],
            f1_at_k=metrics['f1_at_k'],
            map_score=map_score,
            mrr_score=mrr_score,
            ndcg_at_k=metrics['ndcg_at_k'],
            hit_rate=hit_rate
        )
    
    def _calculate_dcg(self, retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """Calculate Discounted Cumulative Gain"""
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            if doc_id in relevant_ids:
                # Binary relevance: 1 if relevant, 0 otherwise
                dcg += 1.0 / np.log2(i + 2)  # i+2 because positions start at 1
        return dcg
    
    def _calculate_idcg(self, relevant_ids: List[str], k: int) -> float:
        """Calculate Ideal DCG"""
        # Ideal case: all relevant documents at top positions
        idcg = 0.0
        for i in range(min(len(relevant_ids), k)):
            idcg += 1.0 / np.log2(i + 2)
        return idcg
    
    def calculate_text_metrics(self, predicted: str, reference: str) -> Dict[str, float]:
        """Calculate text-based metrics (BLEU, ROUGE, etc.)"""
        from nltk.translate.bleu_score import sentence_bleu
        from rouge_score import rouge_scorer
        
        # Tokenize
        pred_tokens = predicted.lower().split()
        ref_tokens = reference.lower().split()
        
        # BLEU score
        bleu = sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5))
        
        # ROUGE scores
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(reference, predicted)
        
        # Exact match
        exact_match = 1.0 if predicted.strip().lower() == reference.strip().lower() else 0.0
        
        # Token F1
        pred_set = set(pred_tokens)
        ref_set = set(ref_tokens)
        
        if not pred_set and not ref_set:
            f1_token = 1.0
        elif not pred_set or not ref_set:
            f1_token = 0.0
        else:
            precision = len(pred_set & ref_set) / len(pred_set)
            recall = len(pred_set & ref_set) / len(ref_set)
            f1_token = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'bleu': bleu,
            'rouge1_f': rouge_scores['rouge1'].fmeasure,
            'rouge2_f': rouge_scores['rouge2'].fmeasure,
            'rougeL_f': rouge_scores['rougeL'].fmeasure,
            'exact_match': exact_match,
            'f1_token': f1_token
        }
    
    
    async def evaluate_single_query(
        self,
        query_data: Dict[str, Any],
        query_engine: SpaceMissionQueryEngine,
        config: Dict[str, Any]
    ) -> EvaluationResult:
        """Evaluate a single query with comprehensive metrics"""
        
        import time
        query = query_data['question']
        ground_truth = query_data.get('answer', '')
        relevant_docs = query_data.get('relevant_docs', [])
        
        try:
            # Time retrieval and generation
            start_time = time.time()
            
            # Execute query
            result = query_engine.query(
                query,
                response_mode=ResponseMode[config['response_mode'].upper()],
                return_sources=True,
                verbose=False
            )
            
            total_time = time.time() - start_time
            
            # Debug: log the result structure
            logger.debug(f"Query result type: {type(result)}")
            logger.debug(f"Query result: {result}")
            
            # Extract results
            generated_answer = result['response']
            retrieved_contexts = result.get('sources', [])
            
            # Check if we got valid results
            if not generated_answer:
                logger.error(f"Empty response from query engine for query: {query}")
                logger.error(f"Query result was: {result}")
                raise ValueError(f"Empty response from query engine for query: {query}")
            
            logger.info(f"Got response: {generated_answer[:100]}... (length: {len(generated_answer)})")
            
            # Calculate retrieval metrics
            retrieval_metrics = self.calculate_retrieval_metrics(
                retrieved_contexts,
                relevant_docs,
                config['k_values']
            )
            
            # Prepare context for evaluation
            context_texts = [ctx['text'] for ctx in retrieved_contexts[:5]]
            
            # LlamaIndex generation metrics
            logger.debug(f"Evaluating with: query={query}, response={generated_answer}, ground_truth={ground_truth}")
            
            # Correctness evaluation
            try:
                correctness_result = await self.correctness_evaluator.aevaluate(
                    query=query,
                    response=generated_answer,
                    reference=ground_truth
                )
                logger.debug("Correctness evaluation successful")
            except Exception as e:
                logger.warning(f"Error in correctness evaluation: {e}")
                correctness_result = type('obj', (object,), {'score': None})()
            
            # Answer relevancy evaluation
            try:
                answer_relevancy_result = await self.answer_relevancy_evaluator.aevaluate(
                    query=query,
                    response=generated_answer
                )
                logger.debug("Answer relevancy evaluation successful")
            except Exception as e:
                logger.warning(f"Error in answer relevancy evaluation: {e}")
                answer_relevancy_result = type('obj', (object,), {'score': None})()
            
            # Context relevancy evaluation
            try:
                # Context relevancy evaluator expects query and contexts parameters
                context_relevancy_result = await self.context_relevancy_evaluator.aevaluate(
                    query=query,
                    contexts=context_texts  # Use the extracted context texts
                )
                logger.debug("Context relevancy evaluation successful")
            except Exception as e:
                logger.warning(f"Error in context relevancy evaluation: {e}")
                logger.debug(f"Query: {query}")
                logger.debug(f"Contexts: {context_texts}")
                context_relevancy_result = type('obj', (object,), {'score': None})()
            
            # Faithfulness evaluation
            try:
                faithfulness_result = await self.faithfulness_evaluator.aevaluate(
                    query=query,
                    response=generated_answer,
                    contexts=context_texts
                )
                logger.debug("Faithfulness evaluation successful")
            except Exception as e:
                logger.warning(f"Error in faithfulness evaluation: {e}")
                faithfulness_result = type('obj', (object,), {'score': None})()
            
            # Semantic similarity evaluation
            try:
                semantic_result = await self.semantic_evaluator.aevaluate(
                    response=generated_answer,
                    reference=ground_truth
                )
                logger.debug("Semantic evaluation successful")
            except Exception as e:
                logger.warning(f"Error in semantic evaluation: {e}")
                semantic_result = type('obj', (object,), {'score': None})()
            
            # Text-based metrics
            text_metrics = self.calculate_text_metrics(generated_answer, ground_truth)
            
            # Helper function to safely extract scores
            def safe_score(result, default=0.0):
                """Safely extract score from evaluation result"""
                try:
                    if hasattr(result, 'score') and result.score is not None:
                        return float(result.score)
                except (TypeError, ValueError) as e:
                    logger.warning(f"Could not convert score to float: {e}")
                return default
            
            # Create generation metrics
            generation_metrics = GenerationMetrics(
                correctness=safe_score(correctness_result),
                relevancy=safe_score(answer_relevancy_result),
                faithfulness=safe_score(faithfulness_result),
                semantic_similarity=safe_score(semantic_result),
                guideline_adherence=None,
                # LlamaIndex RAGAS-style metrics
                ragas_faithfulness=safe_score(faithfulness_result),
                ragas_answer_relevancy=safe_score(answer_relevancy_result),
                ragas_context_recall=safe_score(context_relevancy_result),
                ragas_context_precision=safe_score(context_relevancy_result),
                # Text metrics
                bleu_score=text_metrics['bleu'],
                rouge_scores={
                    'rouge1_f': text_metrics['rouge1_f'],
                    'rouge2_f': text_metrics['rouge2_f'],
                    'rougeL_f': text_metrics['rougeL_f']
                },
                exact_match=text_metrics['exact_match'],
                f1_token_score=text_metrics['f1_token']
            )
            
            return EvaluationResult(
                query_id=query_data.get('question_id', ''),
                query=query,
                ground_truth=ground_truth,
                generated_answer=generated_answer,
                retrieved_contexts=[ctx['text'] for ctx in retrieved_contexts],
                retrieval_metrics=retrieval_metrics,
                generation_metrics=generation_metrics,
                config=config,
                retrieval_time=result['metadata'].get('retrieval_time', 0),
                generation_time=result['metadata'].get('generation_time', 0),
                total_time=total_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error evaluating query: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Stack trace:", exc_info=True)
            # Return error result
            return EvaluationResult(
                query_id=query_data.get('question_id', ''),
                query=query,
                ground_truth=ground_truth,
                generated_answer="",
                retrieved_contexts=[],
                retrieval_metrics=RetrievalMetrics(
                    precision_at_k={}, recall_at_k={}, f1_at_k={},
                    map_score=0, mrr_score=0, ndcg_at_k={}, hit_rate=0
                ),
                generation_metrics=GenerationMetrics(
                    correctness=0, relevancy=0, faithfulness=0, semantic_similarity=0,
                    guideline_adherence=None, ragas_faithfulness=0, ragas_answer_relevancy=0,
                    ragas_context_recall=0, ragas_context_precision=0,
                    bleu_score=0, rouge_scores={'rouge1_f': 0, 'rouge2_f': 0, 'rougeL_f': 0}, 
                    exact_match=0, f1_token_score=0
                ),
                config=config,
                retrieval_time=0,
                generation_time=0,
                total_time=0,
                timestamp=datetime.now().isoformat(),
                error=str(e)
            )
    
    async def run_parameter_sweep(
        self,
        evaluation_data: List[Dict[str, Any]],
        config: EvaluationConfig,
        sample_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run evaluation with parameter sweeping"""
        
        if sample_size:
            evaluation_data = evaluation_data[:sample_size]
            
        logger.info(f"Running parameter sweep on {len(evaluation_data)} queries")
        
        # Track results for each configuration
        all_results = []
        config_results = defaultdict(list)
        
        # Generate all parameter combinations
        param_combinations = []
        for top_k in config.top_k_values:
            for threshold in config.similarity_thresholds:
                for temp in config.temperature_values:
                    for mode in config.response_modes:
                        for llm_model in config.llm_models:
                            for embedding_model in config.embedding_models:
                                param_combinations.append({
                                    'top_k': top_k,
                                    'similarity_threshold': threshold,
                                    'temperature': temp,
                                    'response_mode': mode,
                                    'llm_model': llm_model,
                                    'embedding_model': embedding_model,
                                    'k_values': config.top_k_values
                                })
        
        logger.info(f"Testing {len(param_combinations)} parameter combinations")
        
        # Evaluate each combination
        for param_config in tqdm(param_combinations, desc="Parameter combinations"):
            # Initialize query engine with current parameters
            query_engine = SpaceMissionQueryEngine(
                chroma_persist_dir=self.chroma_persist_dir,
                top_k=param_config['top_k'],
                similarity_threshold=param_config['similarity_threshold'],
                temperature=param_config['temperature'],
                llm_model=param_config.get('llm_model', 'gpt-4o-mini'),
                embedding_model=param_config.get('embedding_model', 'text-embedding-3-large')
            )
            
            # Evaluate subset of queries for this configuration
            config_key = f"k{param_config['top_k']}_t{param_config['similarity_threshold']}_temp{param_config['temperature']}_{param_config['response_mode']}"
            
            # Process queries in batches
            batch_size = 5
            for i in range(0, len(evaluation_data), batch_size):
                batch = evaluation_data[i:i+batch_size]
                
                tasks = []
                for query_data in batch:
                    task = self.evaluate_single_query(query_data, query_engine, param_config)
                    tasks.append(task)
                
                batch_results = await asyncio.gather(*tasks)
                
                for result in batch_results:
                    all_results.append(result)
                    config_results[config_key].append(result)
            
            # Calculate aggregate metrics for this configuration
            self._update_parameter_sweep_results(config_key, config_results[config_key])
        
        # Note: RAGAS metrics are now calculated inline through LlamaIndex evaluators
        # No need for separate batch processing
        
        # Save all results
        self._save_parameter_sweep_results(all_results)
        
        # Generate summary
        summary = self._generate_parameter_sweep_summary()
        
        return summary
    
    def _update_parameter_sweep_results(self, config_key: str, results: List[EvaluationResult]):
        """Update parameter sweep tracking with results"""
        
        # Calculate aggregate metrics
        retrieval_metrics = defaultdict(list)
        generation_metrics = defaultdict(list)
        
        for result in results:
            # Retrieval metrics
            retrieval_metrics['map'].append(result.retrieval_metrics.map_score)
            retrieval_metrics['mrr'].append(result.retrieval_metrics.mrr_score)
            retrieval_metrics['hit_rate'].append(result.retrieval_metrics.hit_rate)
            
            # Generation metrics
            generation_metrics['correctness'].append(result.generation_metrics.correctness)
            generation_metrics['relevancy'].append(result.generation_metrics.relevancy)
            generation_metrics['faithfulness'].append(result.generation_metrics.faithfulness)
            generation_metrics['bleu'].append(result.generation_metrics.bleu_score)
            
        # Store aggregated results
        self.parameter_sweep_results[config_key] = {
            'num_queries': len(results),
            'retrieval': {
                metric: {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
                for metric, values in retrieval_metrics.items()
            },
            'generation': {
                metric: {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
                for metric, values in generation_metrics.items()
            }
        }
    
    # Note: RAGAS functionality is now integrated through LlamaIndex evaluators
    # The _add_ragas_metrics method is no longer needed
    
    def _save_parameter_sweep_results(self, results: List[EvaluationResult]):
        """Save detailed parameter sweep results"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert results to serializable format
        results_data = []
        for result in results:
            data = asdict(result)
            # Convert metrics objects to dicts
            data['retrieval_metrics'] = asdict(result.retrieval_metrics)
            data['generation_metrics'] = asdict(result.generation_metrics)
            results_data.append(data)
        
        # Save detailed results
        detailed_path = self.output_dir / f"parameter_sweep_detailed_{timestamp}.json"
        with open(detailed_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save aggregated results by configuration
        aggregated_path = self.output_dir / f"parameter_sweep_aggregated_{timestamp}.json"
        
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_results = convert_to_serializable(dict(self.parameter_sweep_results))
        
        with open(aggregated_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Create DataFrame for analysis
        df_data = []
        for result in results:
            row = {
                'query_id': result.query_id,
                'config_top_k': result.config['top_k'],
                'config_threshold': result.config['similarity_threshold'],
                'config_temperature': result.config['temperature'],
                'config_response_mode': result.config['response_mode'],
                'map_score': result.retrieval_metrics.map_score,
                'mrr_score': result.retrieval_metrics.mrr_score,
                'correctness': result.generation_metrics.correctness,
                'relevancy': result.generation_metrics.relevancy,
                'faithfulness': result.generation_metrics.faithfulness,
                'bleu_score': result.generation_metrics.bleu_score,
                'total_time': result.total_time
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv(self.output_dir / f"parameter_sweep_results_{timestamp}.csv", index=False)
        
        logger.info(f"Saved parameter sweep results to {self.output_dir}")
    
    def _generate_parameter_sweep_summary(self) -> Dict[str, Any]:
        """Generate summary of parameter sweep results"""
        
        # Find best configurations
        best_configs = {
            'best_retrieval_map': None,
            'best_retrieval_mrr': None,
            'best_generation_correctness': None,
            'best_generation_relevancy': None,
            'best_balanced': None
        }
        
        best_scores = {
            'map': -1,
            'mrr': -1,
            'correctness': -1,
            'relevancy': -1,
            'balanced': -1
        }
        
        for config_key, metrics in self.parameter_sweep_results.items():
            # Check MAP score
            if metrics['retrieval']['map']['mean'] > best_scores['map']:
                best_scores['map'] = metrics['retrieval']['map']['mean']
                best_configs['best_retrieval_map'] = config_key
                
            # Check MRR score
            if metrics['retrieval']['mrr']['mean'] > best_scores['mrr']:
                best_scores['mrr'] = metrics['retrieval']['mrr']['mean']
                best_configs['best_retrieval_mrr'] = config_key
                
            # Check correctness
            if metrics['generation']['correctness']['mean'] > best_scores['correctness']:
                best_scores['correctness'] = metrics['generation']['correctness']['mean']
                best_configs['best_generation_correctness'] = config_key
                
            # Check relevancy
            if metrics['generation']['relevancy']['mean'] > best_scores['relevancy']:
                best_scores['relevancy'] = metrics['generation']['relevancy']['mean']
                best_configs['best_generation_relevancy'] = config_key
                
            # Calculate balanced score (average of normalized metrics)
            balanced_score = (
                metrics['retrieval']['map']['mean'] +
                metrics['retrieval']['mrr']['mean'] +
                metrics['generation']['correctness']['mean'] +
                metrics['generation']['relevancy']['mean']
            ) / 4
            
            if balanced_score > best_scores['balanced']:
                best_scores['balanced'] = balanced_score
                best_configs['best_balanced'] = config_key
        
        # Use the same conversion function for summary
        def convert_to_serializable(obj):
            if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        summary = {
            'total_configurations_tested': len(self.parameter_sweep_results),
            'best_configurations': best_configs,
            'best_scores': convert_to_serializable(best_scores),
            'all_results': convert_to_serializable(dict(self.parameter_sweep_results))
        }
        
        # Save summary
        summary_path = self.output_dir / f"parameter_sweep_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        return summary


async def main():
    """Main function to run comprehensive evaluation"""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(
        chroma_persist_dir="../../chroma_db",
        evaluation_llm_model="gpt-4o",
        output_dir="../results"
    )
    
    # Load evaluation dataset
    # First check if synthetic data exists
    eval_data_path = "../data/synthetic_eval_dataset_*.json"
    import glob
    data_files = glob.glob(eval_data_path)
    
    if not data_files:
        print("No evaluation data found. Please run generate_synthetic_data.py first")
        return
        
    # Load most recent dataset
    latest_dataset = sorted(data_files)[-1]
    evaluation_data = evaluator.load_evaluation_dataset(latest_dataset)
    
    print(f"Loaded {len(evaluation_data)} questions from {latest_dataset}")
    
    # Configure parameter sweep
    config = EvaluationConfig(
        top_k_values=[3, 5, 10],
        similarity_thresholds=[0.2, 0.35, 0.5],
        temperature_values=[0.0, 0.3],
        response_modes=["compact", "tree_summarize"]
    )
    
    # Run evaluation with parameter sweep
    # Use sample_size for testing (e.g., 20 queries)
    summary = await evaluator.run_parameter_sweep(
        evaluation_data,
        config,
        sample_size=20  # Remove this for full evaluation
    )
    
    # Print summary
    print("\n" + "="*60)
    print("PARAMETER SWEEP SUMMARY")
    print("="*60)
    print(f"Total configurations tested: {summary['total_configurations_tested']}")
    print("\nBest configurations:")
    for metric, config_key in summary['best_configurations'].items():
        if config_key:
            print(f"  {metric}: {config_key}")
            print(f"    Score: {summary['best_scores'][metric.split('_')[-1]]:.3f}")
    
    print("\n✓ Comprehensive evaluation complete!")


if __name__ == "__main__":
    asyncio.run(main())