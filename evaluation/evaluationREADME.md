   # RAG Evaluation Framework

A comprehensive evaluation framework for evaluating the Space Mission RAG (Retrieval-Augmented Generation) system. This framework implements state-of-the-art evaluation metrics from LlamaIndex and RAGAS, with parameter sweeping capabilities to find optimal configurations.

## Overview

The evaluation framework consists of three main components:
1. **Data Generation**: Creates synthetic evaluation datasets with diverse question types
2. **Comprehensive Evaluation**: Tests multiple parameter configurations with extensive metrics
3. **Visualization & Reporting**: Generates detailed analysis plots and summary tables

## Key Features

- **Automated Pipeline**: Single command to run the entire evaluation workflow
- **100-Question Evaluation Set**: Generates diverse questions across 9 question types
- **Comprehensive Metrics**: 
  - Retrieval metrics: Precision@K, Recall@K, F1@K, MAP, MRR, NDCG, Hit Rate
  - LlamaIndex metrics: Correctness, Relevancy, Faithfulness, Semantic Similarity, Guideline Adherence
  - RAGAS metrics: Faithfulness, Answer Relevancy, Context Relevancy, Context Recall/Precision, Answer Similarity/Correctness
  - Text metrics: BLEU, ROUGE, Exact Match, Token F1
- **Parameter Sweeping**: Tests multiple configurations to find optimal settings
- **Advanced Visualizations**: 15+ plots including heatmaps, Pareto frontiers, and correlation matrices
- **Cost-Optimized**: Uses GPT-4o-mini by default with configurable models

## Installation

1. Install dependencies:
```bash
cd evaluation
pip install -r requirements.txt
```

2. Ensure you have an OpenAI API key set:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Quick Start

### Run Complete Evaluation Pipeline

The main entry point is `run_evaluation_pipeline.py`, which orchestrates the entire evaluation process:

```bash
# Run with optimised configuration (recommended)
python run_evaluation_pipeline.py --config configs/optimized_config.yaml

# Run a quick test with minimal parameters
python run_evaluation_pipeline.py --quick-test

# Skip data generation if you already have synthetic data
python run_evaluation_pipeline.py --skip-data-generation

# Force regenerate synthetic data
python run_evaluation_pipeline.py --force-regenerate

# Combine options
python run_evaluation_pipeline.py --config configs/optimized_config.yaml --skip-data-generation
```

### Run Individual Components

#### 1. Generate Evaluation Data Only
```bash
cd scripts
python generate_100_questions.py
```

This script generates 100 high-quality evaluation questions distributed across 9 question types:
- **Factual** (20): Direct facts about missions
- **Technical** (20): Specifications and technical details
- **Comparison** (10): Comparing multiple missions
- **Temporal** (10): Time-based questions
- **Analytical** (15): Analysis and reasoning
- **Multi-hop** (10): Questions requiring multiple retrieval steps
- **Numerical** (10): Quantitative questions
- **Causal** (5): Cause-effect relationships

Output files:
- `data/eval_100_questions_[timestamp].json`: Main evaluation dataset
- `data/eval_100_[question_type]_[timestamp].json`: Questions by type
- `data/generation_stats_[timestamp].json`: Generation statistics

#### 2. Run Evaluation Only
```bash
cd scripts
python comprehensive_evaluation.py
```

This script performs parameter sweep evaluation across all configured combinations. It evaluates:
- Multiple retrieval parameters (top_k, similarity_threshold)
- Multiple generation parameters (temperature, response_mode)
- All configured metrics (retrieval, generation, RAGAS)

Output files:
- `results/parameter_sweep_detailed_[timestamp].json`: Raw evaluation results
- `results/parameter_sweep_aggregated_[timestamp].json`: Aggregated metrics
- `results/parameter_sweep_results_[timestamp].csv`: Tabular format
- `results/parameter_sweep_summary_[timestamp].json`: Summary statistics

#### 3. Create Visualizations Only
```bash
cd scripts
python visualization_and_reporting.py <results_timestamp>
# Example: python visualization_and_reporting.py 20240827_044036
```

This script generates comprehensive visualizations from evaluation results:
- **Retrieval Analysis**: Precision/Recall curves, MAP/MRR heatmaps
- **Generation Analysis**: Temperature effects, cost-quality scatter plots
- **Bridging Metrics**: Correlation heatmap between retrieval and generation
- **LlamaIndex & RAGAS**: Metric heatmaps and parameter relationships
- **Best Parameters**: Composite scores and Pareto frontier analysis
- **Summary Tables**: Top configurations and main effects

## Configuration

### Optimised Configuration (`configs/optimized_config.yaml`)

The optimised configuration reduces evaluation time while maintaining comprehensive coverage:

```yaml
# Data Generation Settings
data_generation:
  total_questions: 100
  llm_model: "gpt-4o-mini"
  temperature: 0.1  # Low temperature for consistent questions

# Parameter Sweep Configuration - OPTIMIZED
parameter_sweep:
  top_k_values: [3, 5, 10]  # Most common values
  similarity_thresholds: [0.35, 0.5, 0.7]  # Mid-range values
  temperature_values: [0.0, 0.1, 0.5]  # Key temperature points
  response_modes: ["compact"]  # Fastest mode
  llm_models: ["gpt-4o-mini"]  # Cost-efficient model
  embedding_models: ["text-embedding-3-large"]

# Evaluation Settings
evaluation:
  evaluation_llm_model: "gpt-4o-mini"
  batch_size: 10
  sample_size: 20  # Evaluate subset for speed
```

### Quick Test Configuration
For rapid testing, use the `--quick-test` flag which automatically uses minimal parameters.

## Parameter Sweeping

The framework tests multiple parameter combinations to find optimal settings:

- **Retrieval Parameters**:
  - `top_k`: Number of documents to retrieve [3, 5, 10, 20]
  - `similarity_threshold`: Minimum similarity score [0.0, 0.2, 0.35, 0.5, 0.7]
  
- **Generation Parameters**:
  - `temperature`: LLM temperature [0.0, 0.3, 0.7]
  - `response_mode`: Response synthesis strategy ["compact", "refine", "tree_summarize"]

## Metrics Implemented

### Retrieval Metrics
- **Precision@K**: Fraction of retrieved documents that are relevant
- **Recall@K**: Fraction of relevant documents that are retrieved
- **MAP (Mean Average Precision)**: Average precision across all queries
- **MRR (Mean Reciprocal Rank)**: Reciprocal of the rank of first relevant document
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **Hit Rate**: Fraction of queries with at least one relevant document

### Generation Metrics

#### LlamaIndex Metrics
- **Correctness**: Evaluates factual accuracy of generated answer
- **Relevancy**: Measures if answer addresses the question
- **Faithfulness**: Checks if answer is grounded in retrieved context
- **Semantic Similarity**: Embedding-based similarity with ground truth

#### RAGAS Metrics
- **Context Relevancy**: How relevant retrieved contexts are
- **Answer Relevancy**: How relevant the answer is to question
- **Answer Correctness**: Factual correctness evaluation
- **Context Recall**: Coverage of ground truth by contexts
- **Context Precision**: Precision of retrieved contexts

#### Text-Based Metrics
- **BLEU Score**: N-gram overlap with reference
- **ROUGE Scores**: Recall-oriented understudy for gisting
- **Exact Match**: Exact string matching
- **Token F1**: Token-level F1 score

## Output Structure

```
evaluation/
├── data/                 # Synthetic evaluation datasets
│   ├── eval_100_questions_*.json         # Main dataset
│   ├── eval_100_*_*.json                 # Questions by type
│   └── generation_stats_*.json           # Generation statistics
├── results/              # Evaluation results
│   ├── parameter_sweep_detailed_*.json   # Raw results
│   ├── parameter_sweep_aggregated_*.json # Aggregated metrics
│   ├── parameter_sweep_results_*.csv     # CSV format
│   └── parameter_sweep_summary_*.json    # Summary stats
└── plots/                # Visualizations (15+ plots)
    ├── retrieval_1_topk_vs_recall.pdf
    ├── retrieval_2_topk_vs_precision.pdf
    ├── retrieval_3_map_heatmap.pdf
    ├── retrieval_4_mrr_heatmap.pdf
    ├── generation_5_temperature_effects.pdf
    ├── generation_6_topk_effects.pdf
    ├── generation_7_cost_quality_scatter.pdf
    ├── generation_8_perquery_distributions.pdf
    ├── bridging_9_correlation_heatmap.pdf
    ├── llamaindex_10_metrics_heatmap.pdf
    ├── llamaindex_11_metrics_vs_parameters.pdf
    ├── ragas_12_metrics_heatmap.pdf
    ├── ragas_13_metrics_vs_retrieval.pdf
    ├── best_params_14_composite_scores.pdf
    ├── best_params_15_pareto_frontier.pdf
    ├── table_1_topline_summary.pdf
    ├── table_2_pareto_set.pdf
    ├── table_3_main_effects.pdf
    ├── table_4_metric_definitions.pdf
    └── enhanced_summary_table.csv
```

## Understanding the Results

### Key Visualizations

1. **Retrieval Performance** (plots 1-4):
   - Top-K vs Recall/Precision curves show retrieval effectiveness
   - MAP/MRR heatmaps reveal optimal parameter combinations

2. **Generation Quality** (plots 5-8):
   - Temperature effects on answer quality
   - Cost vs quality trade-offs
   - Per-query metric distributions

3. **Metric Correlations** (plot 9):
   - Shows relationships between retrieval and generation metrics
   - Helps identify which retrieval metrics predict generation quality

4. **Framework-Specific Analysis** (plots 10-13):
   - LlamaIndex and RAGAS metric heatmaps
   - Parameter sensitivity analysis

5. **Optimisation Results** (plots 14-15):
   - Composite scores for all configurations
   - Pareto frontier showing optimal trade-offs

### Summary Tables

- **Table 1**: Top-line metrics summary
- **Table 2**: Pareto-optimal configurations
- **Table 3**: Main effects of each parameter
- **Table 4**: Metric definitions reference
- **Enhanced Summary**: CSV with all configurations ranked

### Key Findings from Our Evaluation

Based on extensive testing, the optimal configuration for the Space Mission RAG system is:
- **top_k**: 5 (balances precision and recall)
- **similarity_threshold**: 0.5 (filters irrelevant documents)
- **temperature**: 0.1 (consistent, focused answers)
- **response_mode**: "compact" (fast and effective)

## Best Practices

1. **Start with Optimised Config**: Use the provided `optimized_config.yaml` for faster evaluation
2. **Monitor API Usage**: 
   - Full evaluation with 100 questions can use ~1000+ API calls
   - Use `sample_size` parameter to limit evaluation scope
   - GPT-4o-mini provides 10x cost savings vs GPT-4
3. **Iterative Testing**:
   - Start with `--quick-test` to verify setup
   - Use small `sample_size` (e.g., 10) for initial runs
   - Gradually increase coverage
4. **Reproducibility**:
   - Save generated datasets for consistent evaluations
   - Use `--skip-data-generation` to reuse existing data
   - Document configuration changes
5. **Performance Optimization**:
   - Adjust `batch_size` based on rate limits
   - Use `max_concurrent_evaluations` for parallel processing
   - Consider local caching for repeated runs

## Extending the Framework

### Adding New Question Types

To add new question types to `generate_100_questions.py`:

1. Update the question distribution:
```python
question_distribution = {
    "factual": 20,
    "technical": 20,
    "your_new_type": 10  # Add here
}
```

2. Add templates for the new type:
```python
def generate_your_new_type_question(mission_data):
    templates = [
        "Your question template about {title}?",
        "Another template using {orbit_regime}?"
    ]
    # Implementation
```

### Adding New Metrics

To add metrics to `comprehensive_evaluation.py`:

1. Update the metrics configuration in your YAML:
```yaml
metrics:
  generation:
    - your_new_metric
```

2. Implement the metric calculation:
```python
# In evaluate_single_query()
if "your_new_metric" in self.config.metrics.generation:
    metrics["your_new_metric"] = self.calculate_your_metric(response, ground_truth)
```

### Custom Visualizations

Add new plots in `visualization_and_reporting.py`:
```python
def create_your_custom_plot(self):
    # Your visualization code
    self.save_plot("your_plot_name.pdf")
```

## Troubleshooting

### Common Issues

1. **"No evaluation data found"**
   - Run data generation first or use existing data with `--skip-data-generation`

2. **API Rate Limits**
   - Reduce `batch_size` in configuration
   - Increase `request_delay_ms`
   - Use smaller `sample_size`

3. **Memory Issues**
   - Process missions in smaller batches
   - Reduce `questions_per_mission`

4. **Missing Visualizations**
   - Ensure matplotlib backend is properly configured
   - Check that all result files exist for the timestamp

## Performance Considerations

### Time Estimates
- **Quick test**: ~5-10 minutes
- **Optimised config (100 questions, 27 combinations)**: ~1-2 hours
- **Full evaluation (100 questions, 180 combinations)**: ~6-8 hours

### Cost Estimates (GPT-4o-mini)
- **Question generation**: ~$0.50-$1.00
- **Evaluation per configuration**: ~$0.10-$0.20
- **Total for optimized config**: ~$3-5
- **Total for full evaluation**: ~$20-30

### Optimisation Tips
- Use `sample_size` to test on a subset of questions
- Start with fewer parameter combinations
- Use `response_mode: ["compact"]` for 3x speed improvement
- Enable `max_concurrent_evaluations` for parallel processing
- Consider caching embeddings for repeated runs

## Recent Updates

- **Optimised Configuration**: Added streamlined config reducing evaluation time by 85%
- **100-Question Generator**: New focused evaluation dataset with 9 question types
- **Enhanced Visualizations**: 15+ plots including Pareto frontier analysis
- **Cost Optimization**: Default to GPT-4o-mini for 10x cost reduction
- **Improved Pipeline**: Single-command evaluation with automatic orchestration

## Author

Emil Ares, 2025

For questions or contributions, please open an issue on the project repository.