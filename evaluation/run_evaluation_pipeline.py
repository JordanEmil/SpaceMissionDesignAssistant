#!/usr/bin/env python3
"""
Main Runner Script for RAG Evaluation Pipeline
Orchestrates the complete evaluation workflow
"""

import os
import sys
import yaml
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime
import subprocess

# Add parent directory to path for query_pipeline import
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Add scripts directory to path
scripts_dir = str(Path(__file__).parent / "scripts")
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from generate_100_questions import Focused100QuestionGenerator
from comprehensive_evaluation import ComprehensiveEvaluator, EvaluationConfig
from visualization_and_reporting import RefinedEvaluationVisualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """Orchestrates the complete RAG evaluation pipeline"""
    
    def __init__(self, config_path: str):
        """Initialize pipeline with configuration"""
        self.config = self._load_config(config_path)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directories
        self._create_directories()
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_directories(self):
        """Create necessary directories"""
        dirs = [
            self.config['system']['output_dir'],
            self.config['system']['plots_dir'],
            self.config['system']['data_dir']
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    async def step1_generate_synthetic_data(self, force_regenerate: bool = False):
        """Step 1: Generate synthetic evaluation data"""
        logger.info("="*60)
        logger.info("STEP 1: GENERATING SYNTHETIC EVALUATION DATA")
        logger.info("="*60)
        
        # Check if data already exists
        data_dir = Path(self.config['system']['data_dir'])
        existing_data = list(data_dir.glob("eval_100_questions_*.json"))
        
        if existing_data and not force_regenerate:
            logger.info(f"Found existing evaluation data: {existing_data[-1]}")
            logger.info("Skipping data generation. Use --force-regenerate to regenerate.")
            return existing_data[-1]
        
        # Generate new 100-question dataset
        generator = Focused100QuestionGenerator(
            documents_dir=self.config['system']['rag_documents_dir'],
            output_dir=self.config['system']['data_dir'],
            llm_model=self.config['data_generation']['llm_model'],
            temperature=self.config['data_generation']['temperature']
        )
        
        # Generate exactly 100 questions
        await generator.generate_100_questions()
        
        # Find the generated dataset
        new_data = sorted(data_dir.glob("eval_100_questions_*.json"))[-1]
        logger.info(f"Generated new dataset: {new_data}")
        
        return new_data
    
    async def step2_run_evaluation(self, dataset_path: Path):
        """Step 2: Run comprehensive evaluation with parameter sweep"""
        logger.info("="*60)
        logger.info("STEP 2: RUNNING COMPREHENSIVE EVALUATION")
        logger.info("="*60)
        
        # Initialize evaluator
        evaluator = ComprehensiveEvaluator(
            chroma_persist_dir=self.config['system']['chroma_persist_dir'],
            evaluation_llm_model=self.config['evaluation']['evaluation_llm_model'],
            output_dir=self.config['system']['output_dir']
        )
        
        # Load evaluation data
        evaluation_data = evaluator.load_evaluation_dataset(str(dataset_path))
        
        # Create evaluation config
        eval_config = EvaluationConfig(
            top_k_values=self.config['parameter_sweep']['top_k_values'],
            similarity_thresholds=self.config['parameter_sweep']['similarity_thresholds'],
            temperature_values=self.config['parameter_sweep']['temperature_values'],
            response_modes=self.config['parameter_sweep']['response_modes'],
            llm_models=self.config['parameter_sweep']['llm_models'],
            embedding_models=self.config['parameter_sweep']['embedding_models']
        )
        
        # Run evaluation
        logger.info(f"Starting evaluation on {len(evaluation_data)} questions")
        summary = await evaluator.run_parameter_sweep(
            evaluation_data,
            eval_config,
            sample_size=self.config['evaluation']['sample_size']
        )
        
        logger.info("Evaluation complete!")
        
        # Return timestamp for visualization step
        # Find the most recent results files
        results_dir = Path(self.config['system']['output_dir'])
        csv_files = sorted(results_dir.glob("parameter_sweep_results_*.csv"))
        if csv_files:
            # Extract timestamp from filename (format: parameter_sweep_results_YYYYMMDD_HHMMSS.csv)
            latest_csv = csv_files[-1]
            # Get the part after 'parameter_sweep_results_'
            timestamp_parts = latest_csv.stem.replace('parameter_sweep_results_', '')
            results_timestamp = timestamp_parts  # This should be YYYYMMDD_HHMMSS
            return results_timestamp
        
        return None
    
    def step3_create_visualizations(self, results_timestamp: str):
        """Step 3: Create visualizations and reports"""
        logger.info("="*60)
        logger.info("STEP 3: CREATING VISUALIZATIONS AND REPORTS")
        logger.info("="*60)
        
        # Initialize visualizer
        visualizer = RefinedEvaluationVisualizer(
            results_dir=self.config['system']['output_dir'],
            plots_dir=self.config['system']['plots_dir']
        )
        
        # File paths
        csv_file = f"{self.config['system']['output_dir']}/parameter_sweep_results_{results_timestamp}.csv"
        detailed_file = f"{self.config['system']['output_dir']}/parameter_sweep_detailed_{results_timestamp}.json"
        summary_file = f"{self.config['system']['output_dir']}/parameter_sweep_summary_{results_timestamp}.json"
        
        # Check if files exist
        if not all(Path(f).exists() for f in [csv_file, detailed_file, summary_file]):
            logger.error(f"Could not find all required files for timestamp {results_timestamp}")
            return False
        
        # Create visualizations
        visualizer.create_all_visualizations(detailed_file, summary_file)
        
        logger.info("Visualizations complete!")
        return True
    
    async def run_pipeline(self, force_regenerate: bool = False, skip_data_generation: bool = False):
        """Run the complete evaluation pipeline"""
        logger.info("Starting RAG Evaluation Pipeline")
        logger.info(f"Configuration: {self.config}")
        logger.info(f"Timestamp: {self.timestamp}")
        
        try:
            # Step 1: Generate synthetic data
            if not skip_data_generation:
                dataset_path = await self.step1_generate_synthetic_data(force_regenerate)
            else:
                # Find existing dataset
                data_dir = Path(self.config['system']['data_dir'])
                
                # First try the new pattern (eval_100_questions)
                existing_100 = sorted(data_dir.glob("eval_100_questions_*.json"))
                if existing_100:
                    dataset_path = existing_100[-1]
                else:
                    # Fallback to old pattern if exists
                    existing_data = sorted(data_dir.glob("synthetic_eval_dataset_*.json"))
                    if existing_data:
                        dataset_path = existing_data[-1]
                    else:
                        logger.error("No existing evaluation data found. Cannot skip data generation.")
                        return
                    
                logger.info(f"Using existing dataset: {dataset_path}")
            
            # Step 2: Run evaluation
            results_timestamp = await self.step2_run_evaluation(dataset_path)
            
            if not results_timestamp:
                logger.error("Evaluation did not produce results files")
                return
            
            # Step 3: Create visualizations
            success = self.step3_create_visualizations(results_timestamp)
            
            if success:
                logger.info("="*60)
                logger.info("PIPELINE COMPLETE!")
                logger.info("="*60)
                logger.info(f"Results saved to: {self.config['system']['output_dir']}")
                logger.info(f"Plots saved to: {self.config['system']['plots_dir']}")
            else:
                logger.error("Failed to create visualizations")
                
        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}")
            raise


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run RAG Evaluation Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Force regeneration of synthetic data even if it exists"
    )
    parser.add_argument(
        "--skip-data-generation",
        action="store_true",
        help="Skip data generation and use existing data"
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Use quick test configuration for rapid testing"
    )
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    # Use quick test config if requested
    if args.quick_test:
        config_path = "configs/quick_test_config.yaml"
    else:
        config_path = args.config
    
    # Initialize and run pipeline
    pipeline = EvaluationPipeline(config_path)
    
    # Run async pipeline
    asyncio.run(pipeline.run_pipeline(
        force_regenerate=args.force_regenerate,
        skip_data_generation=args.skip_data_generation
    ))


if __name__ == "__main__":
    main()