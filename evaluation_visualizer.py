#!/usr/bin/env python3
"""
Evaluation Results Visualizer for RAG System
Creates comprehensive visualizations and reports for evaluation metrics
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Set matplotlib style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class EvaluationVisualizer:
    """Creates visualizations for RAG evaluation results"""
    
    def __init__(
        self,
        results_dir: str = "evaluation_results",
        output_dir: str = "evaluation_reports"
    ):
        """
        Initialize the visualizer
        
        Args:
            results_dir: Directory containing evaluation results
            output_dir: Directory to save visualizations
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Configure plot aesthetics
        self.configure_plot_style()
        
    def configure_plot_style(self):
        """Configure matplotlib style settings"""
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })
    
    def load_latest_results(self) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """Load the most recent evaluation results"""
        # Find latest aggregate metrics file
        aggregate_files = list(self.results_dir.glob("aggregate_metrics_*.json"))
        if not aggregate_files:
            raise FileNotFoundError("No evaluation results found")
        
        latest_aggregate = max(aggregate_files, key=lambda f: f.stat().st_mtime)
        
        # Load aggregate metrics
        with open(latest_aggregate, 'r') as f:
            aggregate_metrics = json.load(f)
        
        # Find corresponding detailed results
        timestamp = latest_aggregate.stem.split('_', 2)[2]
        results_csv = self.results_dir / f"evaluation_results_{timestamp}.csv"
        
        if results_csv.exists():
            detailed_results = pd.read_csv(results_csv)
        else:
            detailed_results = pd.DataFrame()
        
        return aggregate_metrics, detailed_results
    
    def plot_retrieval_metrics(self, metrics: Dict[str, Any]):
        """Create retrieval metrics visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Retrieval Performance Metrics', fontsize=16, y=1.02)
        
        # Precision@k and Recall@k
        ax = axes[0, 0]
        k_values = sorted(metrics['retrieval_metrics']['precision_at_k'].keys())
        precisions = [metrics['retrieval_metrics']['precision_at_k'][k] for k in k_values]
        recalls = [metrics['retrieval_metrics']['recall_at_k'][k] for k in k_values]
        
        x = np.arange(len(k_values))
        width = 0.35
        
        ax.bar(x - width/2, precisions, width, label='Precision@k', alpha=0.8)
        ax.bar(x + width/2, recalls, width, label='Recall@k', alpha=0.8)
        ax.set_xlabel('k')
        ax.set_ylabel('Score')
        ax.set_title('Precision and Recall at Different k Values')
        ax.set_xticks(x)
        ax.set_xticklabels(k_values)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (p, r) in enumerate(zip(precisions, recalls)):
            ax.text(i - width/2, p + 0.01, f'{p:.3f}', ha='center', va='bottom', fontsize=9)
            ax.text(i + width/2, r + 0.01, f'{r:.3f}', ha='center', va='bottom', fontsize=9)
        
        # MAP, MRR, and Hit Rate
        ax = axes[0, 1]
        metrics_names = ['MAP', 'MRR', 'Hit Rate']
        metrics_values = [
            metrics['retrieval_metrics']['mean_map'],
            metrics['retrieval_metrics']['mean_mrr'],
            metrics['retrieval_metrics']['mean_hit_rate']
        ]
        
        bars = ax.bar(metrics_names, metrics_values, alpha=0.8)
        ax.set_ylabel('Score')
        ax.set_title('Overall Retrieval Metrics')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, metrics_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=11)
        
        # Color bars based on performance
        colors = ['green' if v > 0.7 else 'orange' if v > 0.4 else 'red' for v in metrics_values]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
            bar.set_alpha(0.7)
        
        # Precision-Recall Curve
        ax = axes[1, 0]
        if k_values:
            # Create interpolated curve
            k_extended = np.linspace(min(k_values), max(k_values), 100)
            precision_interp = np.interp(k_extended, k_values, precisions)
            recall_interp = np.interp(k_extended, k_values, recalls)
            
            ax.plot(recall_interp, precision_interp, 'b-', linewidth=2, label='P-R Curve')
            ax.scatter(recalls, precisions, color='red', s=50, zorder=5)
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Trade-off')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            
            # Add diagonal reference line
            ax.plot([0, 1], [1, 0], 'k--', alpha=0.3, label='Random baseline')
            ax.legend()
        
        # F1 Score at different k values
        ax = axes[1, 1]
        f1_scores = []
        for k in k_values:
            p = metrics['retrieval_metrics']['precision_at_k'][k]
            r = metrics['retrieval_metrics']['recall_at_k'][k]
            f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
            f1_scores.append(f1)
        
        ax.plot(k_values, f1_scores, 'go-', linewidth=2, markersize=8)
        ax.set_xlabel('k')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Score at Different k Values')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for k, f1 in zip(k_values, f1_scores):
            ax.text(k, f1 + 0.01, f'{f1:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'retrieval_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_generation_metrics(self, metrics: Dict[str, Any], detailed_results: pd.DataFrame):
        """Create generation metrics visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Generation Quality Metrics', fontsize=16, y=1.02)
        
        # Overall generation metrics
        ax = axes[0, 0]
        gen_metrics = metrics['generation_metrics']
        metric_names = ['Exact Match', 'F1 Score', 'Semantic Similarity']
        metric_values = [
            gen_metrics['mean_exact_match'],
            gen_metrics['mean_f1_score'],
            gen_metrics['mean_semantic_similarity']
        ]
        
        bars = ax.bar(metric_names, metric_values, alpha=0.8)
        ax.set_ylabel('Score')
        ax.set_title('Average Generation Metrics')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        
        # Add value labels and color coding
        for bar, value in zip(bars, metric_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=11)
            color = 'green' if value > 0.7 else 'orange' if value > 0.4 else 'red'
            bar.set_color(color)
            bar.set_alpha(0.7)
        
        # Distribution of F1 scores
        if not detailed_results.empty and 'f1_score' in detailed_results.columns:
            ax = axes[0, 1]
            ax.hist(detailed_results['f1_score'], bins=20, alpha=0.7, edgecolor='black')
            ax.axvline(detailed_results['f1_score'].mean(), color='red', linestyle='--',
                      linewidth=2, label=f'Mean: {detailed_results["f1_score"].mean():.3f}')
            ax.set_xlabel('F1 Score')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of F1 Scores')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # LLM-based evaluation metrics (if available)
        ax = axes[1, 0]
        if metrics.get('llm_evaluation_metrics'):
            llm_metrics = metrics['llm_evaluation_metrics']
            if llm_metrics:
                metric_names = list(llm_metrics.keys())
                metric_values = list(llm_metrics.values())
                
                bars = ax.bar([n.replace('mean_', '').title() for n in metric_names], 
                             metric_values, alpha=0.8)
                ax.set_ylabel('Score')
                ax.set_title('LLM-based Evaluation Metrics')
                ax.set_ylim(0, 1.1)
                ax.grid(True, alpha=0.3)
                
                for bar, value in zip(bars, metric_values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=11)
            else:
                ax.text(0.5, 0.5, 'LLM evaluation not performed', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title('LLM-based Evaluation Metrics')
        
        # Semantic similarity distribution
        if not detailed_results.empty and 'semantic_similarity' in detailed_results.columns:
            ax = axes[1, 1]
            ax.hist(detailed_results['semantic_similarity'], bins=20, alpha=0.7, edgecolor='black')
            ax.axvline(detailed_results['semantic_similarity'].mean(), color='red', linestyle='--',
                      linewidth=2, label=f'Mean: {detailed_results["semantic_similarity"].mean():.3f}')
            ax.set_xlabel('Semantic Similarity')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Semantic Similarity Scores')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'generation_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_performance_metrics(self, metrics: Dict[str, Any], detailed_results: pd.DataFrame):
        """Create performance metrics visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('System Performance Metrics', fontsize=16, y=1.02)
        
        # Response time statistics
        ax = axes[0, 0]
        perf_metrics = metrics['performance_metrics']
        times = [
            perf_metrics['mean_response_time'],
            perf_metrics['p95_response_time'],
            perf_metrics['p99_response_time']
        ]
        labels = ['Mean', 'P95', 'P99']
        
        bars = ax.bar(labels, times, alpha=0.8)
        ax.set_ylabel('Response Time (seconds)')
        ax.set_title('Response Time Statistics')
        ax.grid(True, alpha=0.3)
        
        # Add value labels and color coding
        for bar, time in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{time:.2f}s', ha='center', va='bottom', fontsize=11)
            color = 'green' if time < 2 else 'orange' if time < 5 else 'red'
            bar.set_color(color)
            bar.set_alpha(0.7)
        
        # Response time distribution
        if not detailed_results.empty and 'response_time' in detailed_results.columns:
            ax = axes[0, 1]
            ax.hist(detailed_results['response_time'], bins=30, alpha=0.7, edgecolor='black')
            ax.axvline(detailed_results['response_time'].mean(), color='red', linestyle='--',
                      linewidth=2, label=f'Mean: {detailed_results["response_time"].mean():.2f}s')
            ax.axvline(detailed_results['response_time'].quantile(0.95), color='orange', linestyle='--',
                      linewidth=2, label=f'P95: {detailed_results["response_time"].quantile(0.95):.2f}s')
            ax.set_xlabel('Response Time (seconds)')
            ax.set_ylabel('Frequency')
            ax.set_title('Response Time Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Metrics correlation heatmap
        if not detailed_results.empty:
            ax = axes[1, 0]
            numeric_cols = ['f1_score', 'semantic_similarity', 'map_score', 'mrr_score', 'response_time']
            available_cols = [col for col in numeric_cols if col in detailed_results.columns]
            
            if len(available_cols) > 1:
                corr_matrix = detailed_results[available_cols].corr()
                im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
                
                # Add labels
                ax.set_xticks(np.arange(len(available_cols)))
                ax.set_yticks(np.arange(len(available_cols)))
                ax.set_xticklabels([col.replace('_', ' ').title() for col in available_cols], rotation=45, ha='right')
                ax.set_yticklabels([col.replace('_', ' ').title() for col in available_cols])
                
                # Add correlation values
                for i in range(len(available_cols)):
                    for j in range(len(available_cols)):
                        text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                     ha='center', va='center', color='black' if abs(corr_matrix.iloc[i, j]) < 0.5 else 'white')
                
                ax.set_title('Metrics Correlation Matrix')
                plt.colorbar(im, ax=ax)
        
        # Query complexity vs performance
        if not detailed_results.empty and 'query' in detailed_results.columns:
            ax = axes[1, 1]
            # Calculate query complexity (simple: length of query)
            detailed_results['query_length'] = detailed_results['query'].str.len()
            
            scatter = ax.scatter(detailed_results['query_length'], 
                               detailed_results['response_time'],
                               c=detailed_results['f1_score'] if 'f1_score' in detailed_results.columns else 'blue',
                               cmap='viridis', alpha=0.6, s=50)
            
            ax.set_xlabel('Query Length (characters)')
            ax.set_ylabel('Response Time (seconds)')
            ax.set_title('Query Complexity vs Response Time')
            ax.grid(True, alpha=0.3)
            
            if 'f1_score' in detailed_results.columns:
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('F1 Score')
            
            # Add trend line
            if len(detailed_results) > 2:
                z = np.polyfit(detailed_results['query_length'], detailed_results['response_time'], 1)
                p = np.poly1d(z)
                ax.plot(detailed_results['query_length'], p(detailed_results['query_length']), 
                       "r--", alpha=0.8, label=f'Trend: {z[0]:.4f}x + {z[1]:.2f}')
                ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_summary_report(self, metrics: Dict[str, Any]):
        """Create a comprehensive summary report"""
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare summary data
        summary_data = []
        
        # Retrieval metrics
        summary_data.append(['RETRIEVAL METRICS', '', ''])
        summary_data.append(['Metric', 'Value', 'Status'])
        summary_data.append(['Mean Average Precision (MAP)', 
                           f"{metrics['retrieval_metrics']['mean_map']:.3f}",
                           self._get_status(metrics['retrieval_metrics']['mean_map'])])
        summary_data.append(['Mean Reciprocal Rank (MRR)', 
                           f"{metrics['retrieval_metrics']['mean_mrr']:.3f}",
                           self._get_status(metrics['retrieval_metrics']['mean_mrr'])])
        summary_data.append(['Hit Rate', 
                           f"{metrics['retrieval_metrics']['mean_hit_rate']:.3f}",
                           self._get_status(metrics['retrieval_metrics']['mean_hit_rate'])])
        
        # Add best k for precision/recall
        p_at_k = metrics['retrieval_metrics']['precision_at_k']
        best_k = max(p_at_k.keys(), key=lambda k: p_at_k[k])
        summary_data.append(['Best Precision@k', 
                           f"P@{best_k} = {p_at_k[best_k]:.3f}",
                           self._get_status(p_at_k[best_k])])
        
        summary_data.append(['', '', ''])
        
        # Generation metrics
        summary_data.append(['GENERATION METRICS', '', ''])
        summary_data.append(['Metric', 'Value', 'Status'])
        summary_data.append(['Exact Match', 
                           f"{metrics['generation_metrics']['mean_exact_match']:.3f}",
                           self._get_status(metrics['generation_metrics']['mean_exact_match'])])
        summary_data.append(['F1 Score', 
                           f"{metrics['generation_metrics']['mean_f1_score']:.3f}",
                           self._get_status(metrics['generation_metrics']['mean_f1_score'])])
        summary_data.append(['Semantic Similarity', 
                           f"{metrics['generation_metrics']['mean_semantic_similarity']:.3f}",
                           self._get_status(metrics['generation_metrics']['mean_semantic_similarity'])])
        
        summary_data.append(['', '', ''])
        
        # Performance metrics
        summary_data.append(['PERFORMANCE METRICS', '', ''])
        summary_data.append(['Metric', 'Value', 'Status'])
        summary_data.append(['Mean Response Time', 
                           f"{metrics['performance_metrics']['mean_response_time']:.2f}s",
                           self._get_perf_status(metrics['performance_metrics']['mean_response_time'])])
        summary_data.append(['P95 Response Time', 
                           f"{metrics['performance_metrics']['p95_response_time']:.2f}s",
                           self._get_perf_status(metrics['performance_metrics']['p95_response_time'])])
        
        summary_data.append(['', '', ''])
        summary_data.append(['EVALUATION SUMMARY', '', ''])
        summary_data.append(['Total Queries Evaluated', str(metrics['num_queries']), ''])
        summary_data.append(['Timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ''])
        
        # Create table
        table = ax.table(cellText=summary_data,
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.5, 0.3, 0.2])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        # Color code cells
        for i, row in enumerate(summary_data):
            for j, cell in enumerate(row):
                cell_obj = table[(i, j)]
                
                # Headers
                if i == 0 or 'METRICS' in str(row[0]) or 'SUMMARY' in str(row[0]):
                    cell_obj.set_facecolor('#3498db')
                    cell_obj.set_text_props(weight='bold', color='white')
                elif row[0] == 'Metric' or i == 1:
                    cell_obj.set_facecolor('#ecf0f1')
                    cell_obj.set_text_props(weight='bold')
                elif j == 2 and row[2]:  # Status column
                    if row[2] == 'Excellent':
                        cell_obj.set_facecolor('#2ecc71')
                        cell_obj.set_text_props(color='white')
                    elif row[2] == 'Good':
                        cell_obj.set_facecolor('#f39c12')
                        cell_obj.set_text_props(color='white')
                    elif row[2] == 'Poor':
                        cell_obj.set_facecolor('#e74c3c')
                        cell_obj.set_text_props(color='white')
        
        plt.title('RAG System Evaluation Summary Report', fontsize=16, pad=20, weight='bold')
        plt.savefig(self.output_dir / 'summary_report.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _get_status(self, value: float) -> str:
        """Get status label based on metric value"""
        if value >= 0.8:
            return 'Excellent'
        elif value >= 0.6:
            return 'Good'
        else:
            return 'Poor'
    
    def _get_perf_status(self, value: float) -> str:
        """Get status label for performance metrics (lower is better)"""
        if value <= 2:
            return 'Excellent'
        elif value <= 5:
            return 'Good'
        else:
            return 'Poor'
    
    def generate_full_report(self):
        """Generate complete evaluation report with all visualizations"""
        print("Loading evaluation results...")
        try:
            metrics, detailed_results = self.load_latest_results()
        except FileNotFoundError:
            print("No evaluation results found. Please run evaluation_framework.py first.")
            return
        
        print("Generating visualizations...")
        
        # Create all plots
        print("  - Creating retrieval metrics plot...")
        self.plot_retrieval_metrics(metrics)
        
        print("  - Creating generation metrics plot...")
        self.plot_generation_metrics(metrics, detailed_results)
        
        print("  - Creating performance metrics plot...")
        self.plot_performance_metrics(metrics, detailed_results)
        
        print("  - Creating summary report...")
        self.create_summary_report(metrics)
        
        # Create HTML report
        print("  - Creating HTML report...")
        self.create_html_report(metrics)
        
        print(f"\n✓ Evaluation report generated successfully!")
        print(f"✓ Visualizations saved to: {self.output_dir}/")
    
    def create_html_report(self, metrics: Dict[str, Any]):
        """Create an HTML report with all visualizations"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>RAG System Evaluation Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 40px;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        .metric-summary {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin: 30px 0;
        }}
        .metric-card {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
            margin: 10px 0;
        }}
        .metric-label {{
            color: #7f8c8d;
            font-size: 14px;
        }}
        .plot-container {{
            margin: 30px 0;
            text-align: center;
        }}
        .plot-container img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .timestamp {{
            text-align: center;
            color: #7f8c8d;
            margin-top: 40px;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>RAG System Evaluation Report</h1>
        
        <div class="metric-summary">
            <div class="metric-card">
                <div class="metric-label">Mean Average Precision</div>
                <div class="metric-value">{metrics['retrieval_metrics']['mean_map']:.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">F1 Score</div>
                <div class="metric-value">{metrics['generation_metrics']['mean_f1_score']:.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Mean Response Time</div>
                <div class="metric-value">{metrics['performance_metrics']['mean_response_time']:.2f}s</div>
            </div>
        </div>
        
        <h2>Summary Report</h2>
        <div class="plot-container">
            <img src="summary_report.png" alt="Summary Report">
        </div>
        
        <h2>Retrieval Performance</h2>
        <div class="plot-container">
            <img src="retrieval_metrics.png" alt="Retrieval Metrics">
        </div>
        
        <h2>Generation Quality</h2>
        <div class="plot-container">
            <img src="generation_metrics.png" alt="Generation Metrics">
        </div>
        
        <h2>System Performance</h2>
        <div class="plot-container">
            <img src="performance_metrics.png" alt="Performance Metrics">
        </div>
        
        <div class="timestamp">
            Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
"""
        
        with open(self.output_dir / 'evaluation_report.html', 'w') as f:
            f.write(html_content)


def main():
    """Main function to generate evaluation visualizations"""
    visualizer = EvaluationVisualizer()
    visualizer.generate_full_report()


if __name__ == "__main__":
    main()