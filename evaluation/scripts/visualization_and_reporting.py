#!/usr/bin/env python3
"""
Refined Visualization and Reporting Module for RAG Evaluation
Creates minimal, report-ready plots and tables with MATLAB-style theming
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RefinedEvaluationVisualizer:
    """Creates refined visualizations for RAG evaluation results"""
    
    def __init__(self, results_dir: str = "../results", plots_dir: str = "../plots"):
        """Initialize the visualizer"""
        self.results_dir = Path(results_dir)
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up consistent theme
        self.setup_plot_theme()
        
    def setup_plot_theme(self):
        """Sets a consistent theme matching the indexing analysis style"""
        self.palette = {
            "matlab_blue": "#0072BD",
            "matlab_orange": "#D95319",
            "matlab_yellow": "#EDB120",
            "matlab_purple": "#7E2F8E",
            "matlab_green": "#77AC30",
            "matlab_cyan": "#4DBEEE",
            "matlab_red": "#A2142F",
            "text": "#000000",
            "bg": "#FFFFFF",
            "grid": "#E0E0E0"
        }
        
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.family': 'serif', 
            'font.serif': ['Times New Roman', 'Garamond'],
            'axes.labelcolor': self.palette['text'], 
            'axes.titlecolor': self.palette['text'],
            'xtick.color': self.palette['text'], 
            'ytick.color': self.palette['text'],
            'axes.edgecolor': 'black', 
            'axes.linewidth': 1,
            'axes.titlesize': 16, 
            'axes.labelsize': 12,
            'xtick.labelsize': 10, 
            'ytick.labelsize': 10,
            'figure.facecolor': self.palette['bg'], 
            'axes.facecolor': self.palette['bg'],
            'grid.color': self.palette['grid'], 
            'legend.frameon': True,
            'legend.framealpha': 0.9, 
            'legend.facecolor': self.palette['bg'],
            'legend.edgecolor': 'black',
            'legend.fontsize': 10
        })
    
    def load_evaluation_results(self, results_file: str) -> Dict[str, Any]:
        """Load evaluation results from JSON file"""
        with open(results_file, 'r') as f:
            return json.load(f)
    
    def prepare_dataframe_with_ci(self, detailed_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Prepare dataframe with proper normalization and aggregation"""
        data = []
        
        for result in detailed_results:
            # Extract configuration
            config = result['config']
            
            # Extract retrieval metrics (already normalized to [0,1])
            retrieval = result['retrieval_metrics']
            
            # Extract generation metrics
            generation = result['generation_metrics']
            
            # Create row with normalized metrics
            row = {
                # Configuration
                'query_id': result['query_id'],
                'config_top_k': config['top_k'],
                'config_threshold': config['similarity_threshold'],
                'config_temperature': config['temperature'],
                'config_response_mode': config['response_mode'],
                
                # Retrieval metrics (normalized to [0,1])
                'recall_at_k': retrieval['recall_at_k'].get(config['top_k'], 0),
                'precision_at_k': retrieval['precision_at_k'].get(config['top_k'], 0),
                'map_score': retrieval['map_score'],
                'mrr_score': retrieval['mrr_score'],
                
                # Generation metrics
                'exact_match': generation['exact_match'],
                'f1_token_score': generation['f1_token_score'],
                'rouge_l': generation['rouge_scores']['rougeL_f'],
                'bleu_score': generation['bleu_score'],
                
                # LlamaIndex metrics (if available)
                'correctness': generation.get('correctness', 0) / 5.0 if 'correctness' in generation else None,
                'relevancy': generation.get('relevancy'),
                'faithfulness': generation.get('faithfulness'),
                'semantic_similarity': generation.get('semantic_similarity'),
                
                # RAGAS metrics (if available)
                'ragas_faithfulness': generation.get('ragas_faithfulness'),
                'ragas_answer_relevancy': generation.get('ragas_answer_relevancy'),
                'ragas_context_recall': generation.get('ragas_context_recall'),
                'ragas_context_precision': generation.get('ragas_context_precision'),
                
                # Latency
                'retrieval_time': result['retrieval_time'],
                'generation_time': result['generation_time'],
                'total_time': result['total_time']
            }
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def calculate_confidence_interval(self, data: np.array, confidence: float = 0.95) -> Tuple[float, float, float]:
        """Calculate mean and confidence interval using bootstrap"""
        if len(data) == 0:
            return 0, 0, 0
            
        mean = np.mean(data)
        
        # Bootstrap for confidence interval
        n_bootstrap = 1000
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        # Calculate confidence interval
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, alpha/2 * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
        
        return mean, lower, upper
    
    def create_retrieval_plots(self, df: pd.DataFrame):
        """Create the 4 retrieval plots"""
        
        # 1. Top-k vs Recall@k plot
        self.plot_topk_vs_recall(df)
        
        # 2. Top-k vs Precision@k plot
        self.plot_topk_vs_precision(df)
        
        # 3. MAP heatmap
        self.plot_map_heatmap(df)
        
        # 4. MRR heatmap
        self.plot_mrr_heatmap(df)
        
        logger.info("Created all retrieval plots")
    
    def plot_topk_vs_recall(self, df: pd.DataFrame):
        """Plot 1: Top-k vs Recall@k with mean ± 95% CI"""
        # Dynamic figure size based on number of parameters
        n_topk = len(df['config_top_k'].unique())
        n_thresholds = len(df['config_threshold'].unique())
        fig_width = max(10, n_topk * 1.5)
        fig_height = max(6, 6 + (n_thresholds - 2) * 0.5)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Group by top-k and threshold
        top_k_values = sorted(df['config_top_k'].unique())
        thresholds = sorted(df['config_threshold'].unique())
        
        # Dynamic colors for different thresholds
        base_colors = [self.palette['matlab_blue'], self.palette['matlab_orange'], 
                      self.palette['matlab_green'], self.palette['matlab_purple'],
                      self.palette['matlab_yellow'], self.palette['matlab_cyan'],
                      self.palette['matlab_red']]
        # Use modulo to cycle through colors if we have more thresholds than colors
        colors = [base_colors[i % len(base_colors)] for i in range(len(thresholds))]
        
        for i, threshold in enumerate(thresholds):
            means = []
            lower_bounds = []
            upper_bounds = []
            
            for k in top_k_values:
                data = df[(df['config_top_k'] == k) & 
                         (df['config_threshold'] == threshold)]['recall_at_k'].values
                mean, lower, upper = self.calculate_confidence_interval(data)
                means.append(mean)
                lower_bounds.append(lower)
                upper_bounds.append(upper)
            
            # Plot with error bars
            ax.errorbar(top_k_values, means, 
                       yerr=[np.array(means) - np.array(lower_bounds), 
                             np.array(upper_bounds) - np.array(means)],
                       label=f'Threshold={threshold}', 
                       color=colors[i] if i < len(colors) else base_colors[i % len(base_colors)], 
                       marker='o', 
                       markersize=8,
                       linewidth=2,
                       capsize=5)
        
        ax.set_xlabel('Top-k')
        ax.set_ylabel('Recall@k')
        ax.set_title('Top-k vs Recall@k (mean ± 95% CI)', fontsize=16, pad=15)
        ax.set_ylim(0, 1)
        ax.set_xticks(top_k_values)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Add note
        n_queries = df['query_id'].nunique()
        ax.text(0.02, 0.02, f'Recall@k in [0,1]. Macro-averaged across {n_queries} queries.',
                transform=ax.transAxes, fontsize=9, va='bottom', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'retrieval_1_topk_vs_recall.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_topk_vs_precision(self, df: pd.DataFrame):
        """Plot 2: Top-k vs Precision@k with mean ± 95% CI"""
        # Dynamic figure size based on number of parameters
        n_topk = len(df['config_top_k'].unique())
        n_thresholds = len(df['config_threshold'].unique())
        fig_width = max(10, n_topk * 1.5)
        fig_height = max(6, 6 + (n_thresholds - 2) * 0.5)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Group by top-k and threshold
        top_k_values = sorted(df['config_top_k'].unique())
        thresholds = sorted(df['config_threshold'].unique())
        
        # Dynamic colors for different thresholds
        base_colors = [self.palette['matlab_blue'], self.palette['matlab_orange'], 
                      self.palette['matlab_green'], self.palette['matlab_purple'],
                      self.palette['matlab_yellow'], self.palette['matlab_cyan'],
                      self.palette['matlab_red']]
        # Use modulo to cycle through colors if we have more thresholds than colors
        colors = [base_colors[i % len(base_colors)] for i in range(len(thresholds))]
        
        for i, threshold in enumerate(thresholds):
            means = []
            lower_bounds = []
            upper_bounds = []
            
            for k in top_k_values:
                data = df[(df['config_top_k'] == k) & 
                         (df['config_threshold'] == threshold)]['precision_at_k'].values
                mean, lower, upper = self.calculate_confidence_interval(data)
                means.append(mean)
                lower_bounds.append(lower)
                upper_bounds.append(upper)
            
            # Plot with error bars
            ax.errorbar(top_k_values, means, 
                       yerr=[np.array(means) - np.array(lower_bounds), 
                             np.array(upper_bounds) - np.array(means)],
                       label=f'Threshold={threshold}', 
                       color=colors[i] if i < len(colors) else base_colors[i % len(base_colors)], 
                       marker='o', 
                       markersize=8,
                       linewidth=2,
                       capsize=5)
        
        ax.set_xlabel('Top-k')
        ax.set_ylabel('Precision@k')
        ax.set_title('Top-k vs Precision@k (mean ± 95% CI)', fontsize=16, pad=15)
        ax.set_ylim(0, 1)
        ax.set_xticks(top_k_values)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Add note
        n_queries = df['query_id'].nunique()
        ax.text(0.02, 0.02, f'Shows if extra context hurts precision. N={n_queries} queries.',
                transform=ax.transAxes, fontsize=9, va='bottom', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'retrieval_2_topk_vs_precision.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_map_heatmap(self, df: pd.DataFrame):
        """Plot 3: MAP heatmap by (Top-k × Threshold)"""
        # Dynamic figure size based on number of parameters
        n_topk = len(df['config_top_k'].unique())
        n_thresholds = len(df['config_threshold'].unique())
        fig_width = max(8, n_topk * 1.5 + 2)
        fig_height = max(6, n_thresholds * 1.2 + 2)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Create pivot table
        pivot = df.pivot_table(
            values='map_score',
            index='config_threshold',
            columns='config_top_k',
            aggfunc='mean'
        )
        
        # Create heatmap
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='Blues', 
                    cbar_kws={'label': 'MAP Score'},
                    vmin=0, vmax=1, ax=ax,
                    linewidths=0.5, linecolor='black')
        
        ax.set_xlabel('Top-k')
        ax.set_ylabel('Similarity Threshold')
        ax.set_title('MAP by (Top-k × Threshold)', fontsize=16, pad=15)
        
        # Add note
        n_queries = df['query_id'].nunique()
        fig.text(0.5, 0.01, f'MAP values in [0,1]. Averaged across {n_queries} queries.',
                ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'retrieval_3_map_heatmap.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_mrr_heatmap(self, df: pd.DataFrame):
        """Plot 4: MRR heatmap by (Top-k × Threshold)"""
        # Dynamic figure size based on number of parameters
        n_topk = len(df['config_top_k'].unique())
        n_thresholds = len(df['config_threshold'].unique())
        fig_width = max(8, n_topk * 1.5 + 2)
        fig_height = max(6, n_thresholds * 1.2 + 2)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Create pivot table
        pivot = df.pivot_table(
            values='mrr_score',
            index='config_threshold',
            columns='config_top_k',
            aggfunc='mean'
        )
        
        # Create heatmap
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='Blues', 
                    cbar_kws={'label': 'MRR Score'},
                    vmin=0, vmax=1, ax=ax,
                    linewidths=0.5, linecolor='black')
        
        ax.set_xlabel('Top-k')
        ax.set_ylabel('Similarity Threshold')
        ax.set_title('MRR by (Top-k × Threshold)', fontsize=16, pad=15)
        
        # Add note
        n_queries = df['query_id'].nunique()
        fig.text(0.5, 0.01, f'MRR values in [0,1]. Averaged across {n_queries} queries.',
                ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'retrieval_4_mrr_heatmap.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_generation_plots(self, df: pd.DataFrame):
        """Create the 4 generation plots"""
        
        # 5. Temperature main-effect plot
        self.plot_temperature_effects(df)
        
        # 6. Top-k main-effect plot
        self.plot_topk_effects(df)
        
        # 7. Cost-quality scatter plot
        self.plot_cost_quality_scatter(df)
        
        # 8. Per-query distribution plots
        self.plot_perquery_distributions(df)
        
        logger.info("Created all generation plots")
    
    def plot_temperature_effects(self, df: pd.DataFrame):
        """Plot 5: Temperature main-effect on EM/F1/ROUGE-L/BLEU (2×2 small multiples)"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        metrics = ['exact_match', 'f1_token_score', 'rouge_l', 'bleu_score']
        metric_names = ['Exact Match', 'F1 Score', 'ROUGE-L', 'BLEU']
        temperatures = sorted(df['config_temperature'].unique())
        
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx]
            
            # Calculate means and CIs for each temperature
            means = []
            lower_bounds = []
            upper_bounds = []
            
            for temp in temperatures:
                data = df[df['config_temperature'] == temp][metric].values
                mean, lower, upper = self.calculate_confidence_interval(data)
                means.append(mean)
                lower_bounds.append(lower)
                upper_bounds.append(upper)
            
            # Plot bars with error bars
            x_pos = np.arange(len(temperatures))
            bars = ax.bar(x_pos, means, color=self.palette['matlab_blue'], alpha=0.8, width=0.6)
            ax.errorbar(x_pos, means,
                       yerr=[np.array(means) - np.array(lower_bounds),
                             np.array(upper_bounds) - np.array(means)],
                       fmt='none', color='black', capsize=5)
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f'{t}' for t in temperatures])
            ax.set_xlabel('Temperature')
            ax.set_ylabel(name)
            ax.set_ylim(0, 1)
            ax.grid(True, axis='y', alpha=0.3)
            ax.set_title(name, fontsize=12)
        
        plt.suptitle('Temperature Main-Effect on Generation Metrics (mean ± 95% CI)', 
                     fontsize=16, y=0.98)
        
        # Add note
        fig.text(0.5, 0.01, 
                'BLEU/ROUGE may saturate on short factual answers—EM/F1 are primary.',
                ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'generation_5_temperature_effects.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_topk_effects(self, df: pd.DataFrame):
        """Plot 6: Top-k main-effect on EM/F1/ROUGE-L/BLEU (2×2 small multiples)"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        metrics = ['exact_match', 'f1_token_score', 'rouge_l', 'bleu_score']
        metric_names = ['Exact Match', 'F1 Score', 'ROUGE-L', 'BLEU']
        top_k_values = sorted(df['config_top_k'].unique())
        thresholds = sorted(df['config_threshold'].unique())
        
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx]
            
            # Dynamic bar width based on number of thresholds
            n_thresholds = len(thresholds)
            x_pos = np.arange(len(top_k_values))
            # Adjust width to prevent overlap
            width = 0.8 / n_thresholds  # Total width divided by number of groups
            
            for i, threshold in enumerate(thresholds):
                means = []
                lower_bounds = []
                upper_bounds = []
                
                for k in top_k_values:
                    data = df[(df['config_top_k'] == k) & 
                             (df['config_threshold'] == threshold)][metric].values
                    mean, lower, upper = self.calculate_confidence_interval(data)
                    means.append(mean)
                    lower_bounds.append(lower)
                    upper_bounds.append(upper)
                
                # Plot bars with error bars
                # Center the bars around each x position
                offset = width * (i - (n_thresholds - 1) / 2)
                # Dynamic color selection
                base_colors = [self.palette['matlab_blue'], self.palette['matlab_orange'], 
                              self.palette['matlab_green'], self.palette['matlab_purple'],
                              self.palette['matlab_yellow'], self.palette['matlab_cyan'],
                              self.palette['matlab_red']]
                color = base_colors[i % len(base_colors)]
                bars = ax.bar(x_pos + offset, means, width, 
                             label=f'Threshold={threshold}', color=color, alpha=0.8)
                ax.errorbar(x_pos + offset, means,
                           yerr=[np.array(means) - np.array(lower_bounds),
                                 np.array(upper_bounds) - np.array(means)],
                           fmt='none', color='black', capsize=3)
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f'{k}' for k in top_k_values])
            ax.set_xlabel('Top-k')
            ax.set_ylabel(name)
            ax.set_ylim(0, 1)
            ax.grid(True, axis='y', alpha=0.3)
            ax.set_title(name, fontsize=12)
            if idx == 0:
                ax.legend(loc='upper left', fontsize=9)
        
        plt.suptitle('Top-k Main-Effect on Generation Metrics (mean ± 95% CI)', 
                     fontsize=16, y=0.98)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'generation_6_topk_effects.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_cost_quality_scatter(self, df: pd.DataFrame):
        """Plot 7: Cost–quality scatter (Generation)"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Group by configuration
        config_data = df.groupby(['config_top_k', 'config_threshold', 
                                  'config_temperature', 'config_response_mode']).agg({
            'f1_token_score': 'mean',
            'exact_match': 'mean',
            'total_time': 'mean'
        }).reset_index()
        
        # Create scatter plot
        scatter = ax.scatter(config_data['total_time'], 
                           config_data['f1_token_score'],
                           c=config_data['config_temperature'],
                           s=config_data['exact_match'] * 300,  # Size by EM
                           alpha=0.7,
                           cmap='coolwarm',
                           edgecolors='black',
                           linewidth=1)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Temperature', rotation=270, labelpad=20)
        
        # Label top 3 points by F1
        top_3 = config_data.nlargest(3, 'f1_token_score')
        for _, row in top_3.iterrows():
            label = f"k={row['config_top_k']},t={row['config_threshold']:.2f}"
            ax.annotate(label, 
                       (row['total_time'], row['f1_token_score']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
        
        ax.set_xlabel('Total Latency (s)')
        ax.set_ylabel('F1 Score')
        ax.set_title('Cost–Quality Scatter (Generation)', fontsize=16, pad=15)
        ax.grid(True, alpha=0.3)
        
        # Add legend for size
        legend_sizes = [0.2, 0.5, 0.8]
        legend_elements = [plt.scatter([], [], s=s*300, c='gray', alpha=0.7, 
                                      edgecolors='black', linewidth=1,
                                      label=f'EM={s}') for s in legend_sizes]
        legend = ax.legend(handles=legend_elements, loc='lower right', 
                          title='Exact Match', frameon=True)
        
        # Add note
        ax.text(0.02, 0.98, 'Pareto view: pick by latency budget.',
                transform=ax.transAxes, fontsize=9, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='black', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'generation_7_cost_quality_scatter.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_perquery_distributions(self, df: pd.DataFrame):
        """Plot 8: Per-query distribution for 2–3 finalist configs"""
        # Select top 3 configurations based on F1 score
        config_scores = df.groupby(['config_top_k', 'config_threshold', 
                                    'config_temperature', 'config_response_mode']).agg({
            'f1_token_score': 'mean'
        }).reset_index()
        
        top_configs = config_scores.nlargest(3, 'f1_token_score')
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Prepare data for each config
        config_data = []
        config_labels = []
        
        for idx, (_, row) in enumerate(top_configs.iterrows()):
            mask = (df['config_top_k'] == row['config_top_k']) & \
                   (df['config_threshold'] == row['config_threshold']) & \
                   (df['config_temperature'] == row['config_temperature']) & \
                   (df['config_response_mode'] == row['config_response_mode'])
            
            config_subset = df[mask]
            config_data.append(config_subset)
            label = f"Config {chr(65+idx)}: k={row['config_top_k']}, " + \
                   f"t={row['config_threshold']:.2f}, T={row['config_temperature']}"
            config_labels.append(label)
        
        # Plot EM distributions
        em_data = [config['exact_match'].values for config in config_data]
        positions = np.arange(len(config_labels))
        
        bp1 = ax1.boxplot(em_data, positions=positions, widths=0.6,
                          patch_artist=True, showmeans=True,
                          meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
        
        # Color the boxes
        colors = [self.palette['matlab_blue'], self.palette['matlab_orange'], self.palette['matlab_green']]
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_xticklabels(config_labels, rotation=15, ha='right')
        ax1.set_ylabel('Exact Match Score')
        ax1.set_title('Exact Match Distribution by Configuration', fontsize=14)
        ax1.grid(True, axis='y', alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Plot F1 distributions
        f1_data = [config['f1_token_score'].values for config in config_data]
        
        bp2 = ax2.boxplot(f1_data, positions=positions, widths=0.6,
                          patch_artist=True, showmeans=True,
                          meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
        
        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_xticklabels(config_labels, rotation=15, ha='right')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('F1 Score Distribution by Configuration', fontsize=14)
        ax2.grid(True, axis='y', alpha=0.3)
        ax2.set_ylim(0, 1)
        
        plt.suptitle('Per-Query Distribution for Top 3 Configurations', fontsize=16)
        
        # Add note
        fig.text(0.5, 0.01, 'Shows robustness and outliers. Red diamonds indicate means.',
                ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'generation_8_perquery_distributions.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_bridging_plot(self, df: pd.DataFrame):
        """Create the bridging plot (correlation heatmap)"""
        
        # Plot 9: Correlation heatmap
        self.plot_correlation_heatmap(df)
        
        logger.info("Created bridging plot")
    
    def plot_correlation_heatmap(self, df: pd.DataFrame):
        """Plot 9: Correlation heatmap (Retrieval ↔ Generation metrics)"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Select metrics - include LlamaIndex and RAGAS if available
        base_metrics = ['recall_at_k', 'precision_at_k', 'map_score', 'mrr_score',
                       'exact_match', 'f1_token_score', 'rouge_l', 'bleu_score']
        llamaindex_metrics = ['correctness', 'relevancy', 'faithfulness', 'semantic_similarity']
        ragas_metrics = ['ragas_faithfulness', 'ragas_answer_relevancy', 
                        'ragas_context_recall', 'ragas_context_precision']
        
        # Check which metrics are available
        available_metrics = []
        metric_names = []
        
        for metric, name in zip(base_metrics, ['Recall@k', 'Precision@k', 'MAP', 'MRR',
                                               'EM', 'F1', 'ROUGE-L', 'BLEU']):
            if metric in df.columns and df[metric].notna().any():
                available_metrics.append(metric)
                metric_names.append(name)
        
        # Add LlamaIndex metrics if available
        for metric, name in zip(llamaindex_metrics, ['Correct', 'Relev', 'Faith', 'SemSim']):
            if metric in df.columns and df[metric].notna().any():
                available_metrics.append(metric)
                metric_names.append(name)
        
        # Add RAGAS metrics if available
        for metric, name in zip(ragas_metrics, ['R-Faith', 'R-AnsRel', 'R-CtxRec', 'R-CtxPre']):
            if metric in df.columns and df[metric].notna().any():
                available_metrics.append(metric)
                metric_names.append(name)
        
        # Calculate correlations
        corr_matrix = df[available_metrics].corr()
        
        # Create heatmap
        mask = np.zeros_like(corr_matrix)
        mask[np.triu_indices_from(mask, k=1)] = True
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                    vmin=-1, vmax=1, center=0, square=True,
                    xticklabels=metric_names, yticklabels=metric_names,
                    mask=mask, ax=ax,
                    linewidths=0.5, linecolor='black',
                    cbar_kws={'label': 'Correlation Coefficient'})
        
        ax.set_title('Correlation Heatmap (All Available Metrics)', 
                    fontsize=16, pad=15)
        
        # Add note
        fig.text(0.5, 0.01, 
                'Shows correlations between retrieval, generation, and evaluation metrics.',
                ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'bridging_9_correlation_heatmap.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_tables(self, df: pd.DataFrame):
        """Create all tables"""
        
        # Table 1: Top-line summary
        self.create_topline_summary_table(df)
        
        # Table 2: Pareto set
        self.create_pareto_set_table(df)
        
        # Table 3: Main effects
        self.create_main_effects_table(df)
        
        # Table 4: Metric definitions
        self.create_metric_definitions_table()
        
        logger.info("Created all tables")
    
    def create_topline_summary_table(self, df: pd.DataFrame):
        """Table 1: Top-line summary sorted by MAP"""
        
        # Aggregate by configuration
        config_cols = ['config_top_k', 'config_threshold', 'config_temperature', 'config_response_mode']
        
        summary_data = []
        
        for name, group in df.groupby(config_cols):
            metrics = {}
            
            # Calculate mean and CI for each metric
            for metric in ['recall_at_k', 'precision_at_k', 'map_score', 'mrr_score',
                          'exact_match', 'f1_token_score', 'rouge_l', 'bleu_score']:
                mean, lower, upper = self.calculate_confidence_interval(group[metric].values)
                metrics[f'{metric}_mean'] = mean
                metrics[f'{metric}_ci'] = f"±{(upper - lower) / 2:.3f}"
            
            # Calculate latency percentiles
            metrics['latency_p50'] = group['total_time'].quantile(0.5)
            metrics['latency_p95'] = group['total_time'].quantile(0.95)
            
            # Add configuration
            metrics.update({
                'Config ID': f"{name[0]}-{name[1]:.2f}-{name[2]}-{name[3][:4]}",
                'Top-k': name[0],
                'Threshold': name[1],
                'Temp': name[2],
                'Response Mode': name[3]
            })
            
            summary_data.append(metrics)
        
        # Create DataFrame and sort by MAP
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('map_score_mean', ascending=False).head(10)
        
        # Format table for display - dynamic size based on content
        n_rows = min(10, len(summary_df))
        fig_height = max(8, n_rows * 0.8 + 2)
        fig, ax = plt.subplots(figsize=(18, fig_height))
        ax.axis('off')
        
        # Prepare table data
        table_data = []
        headers = ['Config ID', 'Top-k', 'Threshold', 'Temp', 'Mode',
                  'Recall@k', 'Precision@k', 'MAP', 'MRR',
                  'EM', 'F1', 'ROUGE-L', 'BLEU',
                  'Latency p50/p95 (s)']
        
        for _, row in summary_df.iterrows():
            table_row = [
                row['Config ID'],
                str(row['Top-k']),
                f"{row['Threshold']:.2f}",
                f"{row['Temp']:.1f}",
                row['Response Mode'][:8],
                f"{row['recall_at_k_mean']:.3f}{row['recall_at_k_ci']}",
                f"{row['precision_at_k_mean']:.3f}{row['precision_at_k_ci']}",
                f"{row['map_score_mean']:.3f}{row['map_score_ci']}",
                f"{row['mrr_score_mean']:.3f}{row['mrr_score_ci']}",
                f"{row['exact_match_mean']:.3f}{row['exact_match_ci']}",
                f"{row['f1_token_score_mean']:.3f}{row['f1_token_score_ci']}",
                f"{row['rouge_l_mean']:.3f}{row['rouge_l_ci']}",
                f"{row['bleu_score_mean']:.3f}{row['bleu_score_ci']}",
                f"{row['latency_p50']:.2f}/{row['latency_p95']:.2f}"
            ]
            table_data.append(table_row)
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center')
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.5)
        
        # Color header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor(self.palette['matlab_blue'])
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Highlight best values
        for col_idx, metric in enumerate(['recall_at_k_mean', 'precision_at_k_mean', 
                                         'map_score_mean', 'mrr_score_mean',
                                         'exact_match_mean', 'f1_token_score_mean']):
            if metric in summary_df.columns:
                best_idx = summary_df[metric].idxmax()
                if best_idx in summary_df.index:
                    row_idx = summary_df.index.get_loc(best_idx) + 1
                    col_map = {'recall_at_k_mean': 5, 'precision_at_k_mean': 6,
                              'map_score_mean': 7, 'mrr_score_mean': 8,
                              'exact_match_mean': 9, 'f1_token_score_mean': 10}
                    if metric in col_map:
                        table[(row_idx, col_map[metric])].set_facecolor('#90EE90')
        
        plt.title('Table 1: Top-Line Summary (Sorted by MAP)', fontsize=16, pad=20)
        
        # Add note
        fig.text(0.5, 0.05, 'Values show mean ± 95% CI. Best values highlighted in green.',
                ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'table_1_topline_summary.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_pareto_set_table(self, df: pd.DataFrame):
        """Table 2: Pareto set (quality vs cost)"""
        
        # Aggregate by configuration
        config_cols = ['config_top_k', 'config_threshold', 'config_temperature', 'config_response_mode']
        
        config_summary = df.groupby(config_cols).agg({
            'f1_token_score': 'mean',
            'exact_match': 'mean',
            'map_score': 'mean',
            'mrr_score': 'mean',
            'total_time': ['mean', 'quantile']
        }).reset_index()
        
        # Flatten column names
        config_summary.columns = ['_'.join(col).strip() if col[1] else col[0] 
                                 for col in config_summary.columns.values]
        config_summary.rename(columns={'total_time_quantile': 'latency_p50'}, inplace=True)
        
        # Find Pareto frontier
        pareto_configs = []
        
        for idx, row in config_summary.iterrows():
            is_pareto = True
            for _, other in config_summary.iterrows():
                if (other['f1_token_score_mean'] > row['f1_token_score_mean'] and 
                    other['latency_p50'] < row['latency_p50']):
                    is_pareto = False
                    break
            if is_pareto:
                pareto_configs.append(idx)
        
        pareto_df = config_summary.iloc[pareto_configs].sort_values('latency_p50')
        
        # Create table - dynamic size based on content
        n_rows = len(pareto_df)
        fig_height = max(6, n_rows * 0.8 + 2)
        fig, ax = plt.subplots(figsize=(12, fig_height))
        ax.axis('off')
        
        # Prepare table data
        headers = ['Config ID', 'MAP', 'MRR', 'F1', 'EM', 'Latency p50', 'Note']
        table_data = []
        
        for idx, (_, row) in enumerate(pareto_df.iterrows()):
            config_id = f"{row['config_top_k']}-{row['config_threshold']:.2f}-" + \
                       f"{row['config_temperature']}-{row['config_response_mode'][:4]}"
            
            # Determine note
            if idx == 0:
                note = "Fastest acceptable"
            elif idx == len(pareto_df) - 1:
                note = "Best quality"
            else:
                note = "Balanced"
            
            table_row = [
                config_id,
                f"{row['map_score_mean']:.3f}",
                f"{row['mrr_score_mean']:.3f}",
                f"{row['f1_token_score_mean']:.3f}",
                f"{row['exact_match_mean']:.3f}",
                f"{row['latency_p50']:.2f}s",
                note
            ]
            table_data.append(table_row)
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center')
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        
        # Color header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor(self.palette['matlab_blue'])
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('Table 2: Pareto Set (Quality vs Cost)', fontsize=16, pad=20)
        
        # Add note
        fig.text(0.5, 0.1, 'Only configurations on the F1-Latency Pareto frontier.',
                ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'table_2_pareto_set.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_main_effects_table(self, df: pd.DataFrame):
        """Table 3: Main-effects (Δ vs baseline)"""
        
        # Define baseline configuration
        baseline_config = {
            'config_top_k': 5,
            'config_threshold': 0.35,
            'config_temperature': 0.0,
            'config_response_mode': df['config_response_mode'].mode()[0]
        }
        
        # Get baseline metrics
        baseline_mask = True
        for key, value in baseline_config.items():
            baseline_mask &= (df[key] == value)
        
        # Get numeric columns only
        numeric_cols = ['recall_at_k', 'precision_at_k', 'map_score', 'mrr_score',
                       'exact_match', 'f1_token_score', 'rouge_l', 'bleu_score',
                       'retrieval_time', 'generation_time', 'total_time']
        baseline_metrics = df[baseline_mask][numeric_cols].mean()
        
        # Calculate main effects
        effects_data = []
        
        # Top-k effect
        for k in [10]:
            mask = (df['config_top_k'] == k)
            for key, value in baseline_config.items():
                if key != 'config_top_k':
                    mask &= (df[key] == value)
            
            if mask.sum() > 0:
                deltas = df[mask][numeric_cols].mean() - baseline_metrics
                effects_data.append({
                    'Factor Change': f'Top-k: 5→{k}',
                    'ΔMAP': f"{deltas['map_score']:.3f}±{self._get_ci_width(df[mask]['map_score'].values):.3f}",
                    'ΔMRR': f"{deltas['mrr_score']:.3f}±{self._get_ci_width(df[mask]['mrr_score'].values):.3f}",
                    'ΔEM': f"{deltas['exact_match']:.3f}±{self._get_ci_width(df[mask]['exact_match'].values):.3f}",
                    'ΔF1': f"{deltas['f1_token_score']:.3f}±{self._get_ci_width(df[mask]['f1_token_score'].values):.3f}",
                    'ΔLatency': f"{deltas['total_time']:.2f}±{self._get_ci_width(df[mask]['total_time'].values):.2f}"
                })
        
        # Threshold effect
        for t in [0.5]:
            mask = (df['config_threshold'] == t)
            for key, value in baseline_config.items():
                if key != 'config_threshold':
                    mask &= (df[key] == value)
            
            if mask.sum() > 0:
                deltas = df[mask][numeric_cols].mean() - baseline_metrics
                effects_data.append({
                    'Factor Change': f'Threshold: 0.35→{t}',
                    'ΔMAP': f"{deltas['map_score']:.3f}±{self._get_ci_width(df[mask]['map_score'].values):.3f}",
                    'ΔMRR': f"{deltas['mrr_score']:.3f}±{self._get_ci_width(df[mask]['mrr_score'].values):.3f}",
                    'ΔEM': f"{deltas['exact_match']:.3f}±{self._get_ci_width(df[mask]['exact_match'].values):.3f}",
                    'ΔF1': f"{deltas['f1_token_score']:.3f}±{self._get_ci_width(df[mask]['f1_token_score'].values):.3f}",
                    'ΔLatency': f"{deltas['total_time']:.2f}±{self._get_ci_width(df[mask]['total_time'].values):.2f}"
                })
        
        # Temperature effect
        for temp in [0.1]:
            mask = (df['config_temperature'] == temp)
            for key, value in baseline_config.items():
                if key != 'config_temperature':
                    mask &= (df[key] == value)
            
            if mask.sum() > 0:
                deltas = df[mask][numeric_cols].mean() - baseline_metrics
                effects_data.append({
                    'Factor Change': f'Temp: 0.0→{temp}',
                    'ΔMAP': f"{deltas['map_score']:.3f}±{self._get_ci_width(df[mask]['map_score'].values):.3f}",
                    'ΔMRR': f"{deltas['mrr_score']:.3f}±{self._get_ci_width(df[mask]['mrr_score'].values):.3f}",
                    'ΔEM': f"{deltas['exact_match']:.3f}±{self._get_ci_width(df[mask]['exact_match'].values):.3f}",
                    'ΔF1': f"{deltas['f1_token_score']:.3f}±{self._get_ci_width(df[mask]['f1_token_score'].values):.3f}",
                    'ΔLatency': f"{deltas['total_time']:.2f}±{self._get_ci_width(df[mask]['total_time'].values):.2f}"
                })
        
        # Create table - dynamic size based on content
        n_rows = len(effects_data)
        fig_height = max(4, n_rows * 0.6 + 2)
        fig, ax = plt.subplots(figsize=(10, fig_height))
        ax.axis('off')
        
        # Check if we have any effects data
        if not effects_data:
            ax.text(0.5, 0.5, 'No main effects data available\n(Baseline configuration may not be present in results)',
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            plt.title('Table 3: Main Effects (Δ vs Baseline)', fontsize=16, pad=20)
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'table_3_main_effects.pdf', dpi=300, bbox_inches='tight')
            plt.close()
            return
            
        # Prepare table data
        headers = list(effects_data[0].keys())
        table_data = [[effect[col] for col in headers] for effect in effects_data]
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center')
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Color header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor(self.palette['matlab_blue'])
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('Table 3: Main Effects (Δ vs Baseline)', fontsize=16, pad=20)
        
        # Add note
        fig.text(0.5, 0.15, f'Baseline: k={baseline_config["config_top_k"]}, '
                           f't={baseline_config["config_threshold"]}, '
                           f'T={baseline_config["config_temperature"]}',
                ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'table_3_main_effects.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _get_ci_width(self, data: np.array) -> float:
        """Helper to get CI half-width"""
        _, lower, upper = self.calculate_confidence_interval(data)
        return (upper - lower) / 2
    
    def create_metric_definitions_table(self):
        """Table 4: Metric definitions & evaluation settings"""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        # Define metrics
        metric_defs = [
            ['Metric', 'Range', 'Averaging', 'Cutoffs', 'Note'],
            ['Recall@k', '[0,1]', 'Macro over queries', 'k=5,10', 'Fraction of relevant docs retrieved'],
            ['Precision@k', '[0,1]', 'Macro over queries', 'k=5,10', 'Fraction of retrieved docs that are relevant'],
            ['MAP', '[0,1]', 'Macro over queries', 'All ranks', 'Mean Average Precision across queries'],
            ['MRR', '[0,1]', 'Macro over queries', 'All ranks', 'Mean Reciprocal Rank of first relevant doc'],
            ['Exact Match', '[0,1]', 'Macro over queries', 'N/A', 'Binary: 1 if generated = ground truth'],
            ['F1 Score', '[0,1]', 'Macro over queries', 'N/A', 'Token-level F1 between generated and ground truth'],
            ['ROUGE-L', '[0,1]', 'Macro over queries', 'N/A', 'Longest common subsequence F-score'],
            ['BLEU', '[0,1]', 'Macro over queries', 'N/A', 'Modified unigram precision (may saturate on short answers)'],
            ['Latency', '[0,∞) sec', 'Percentiles', 'p50, p95', 'Total time including retrieval and generation']
        ]
        
        # Create table
        table = ax.table(cellText=metric_defs[1:], colLabels=metric_defs[0],
                        cellLoc='left', loc='center')
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 2)
        
        # Color header
        for i in range(len(metric_defs[0])):
            table[(0, i)].set_facecolor(self.palette['matlab_blue'])
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(metric_defs)):
            if i % 2 == 0:
                for j in range(len(metric_defs[0])):
                    table[(i, j)].set_facecolor('#F0F0F0')
        
        plt.title('Table 4: Metric Definitions & Evaluation Settings', fontsize=16, pad=20)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'table_4_metric_definitions.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    
    def create_llamaindex_plots(self, df: pd.DataFrame):
        """Create plots for LlamaIndex metrics if available"""
        # Check if LlamaIndex metrics are available
        llamaindex_metrics = ['correctness', 'relevancy', 'faithfulness', 'semantic_similarity']
        if not any(col in df.columns and df[col].notna().any() for col in llamaindex_metrics):
            logger.info("No LlamaIndex metrics found, skipping LlamaIndex plots")
            return
        
        logger.info("Creating LlamaIndex plots...")
        
        # Plot 10: LlamaIndex metrics heatmap
        self.plot_llamaindex_heatmap(df)
        
        # Plot 11: LlamaIndex metrics vs parameters
        self.plot_llamaindex_vs_parameters(df)
    
    def plot_llamaindex_heatmap(self, df: pd.DataFrame):
        """Plot 10: LlamaIndex metrics heatmap by configuration"""
        # Filter to only rows with LlamaIndex metrics
        llamaindex_metrics = ['correctness', 'relevancy', 'faithfulness', 'semantic_similarity']
        df_llama = df[df[llamaindex_metrics].notna().all(axis=1)]
        
        if df_llama.empty:
            return
        
        # Aggregate by configuration
        config_cols = ['config_top_k', 'config_threshold', 'config_temperature', 'config_response_mode']
        agg_data = df_llama.groupby(config_cols)[llamaindex_metrics].mean().reset_index()
        
        # Create configuration labels
        agg_data['config_label'] = agg_data.apply(
            lambda row: f"k={row['config_top_k']},t={row['config_threshold']:.1f},T={row['config_temperature']:.1f}",
            axis=1
        )
        
        # Select top 15 configurations by average score
        agg_data['avg_score'] = agg_data[llamaindex_metrics].mean(axis=1)
        top_configs = agg_data.nlargest(15, 'avg_score')
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data for heatmap
        heatmap_data = top_configs[llamaindex_metrics].T
        heatmap_data.columns = top_configs['config_label']
        heatmap_data.index = ['Correctness', 'Relevancy', 'Faithfulness', 'Semantic Similarity']
        
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd',
                    vmin=0, vmax=1, ax=ax,
                    cbar_kws={'label': 'Metric Score'},
                    linewidths=0.5, linecolor='black')
        
        ax.set_xlabel('Configuration')
        ax.set_ylabel('LlamaIndex Metric')
        ax.set_title('LlamaIndex Metrics by Configuration (Top 15)', fontsize=16, pad=15)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'llamaindex_10_metrics_heatmap.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_llamaindex_vs_parameters(self, df: pd.DataFrame):
        """Plot 11: LlamaIndex metrics vs parameters"""
        llamaindex_metrics = ['correctness', 'relevancy', 'faithfulness', 'semantic_similarity']
        df_llama = df[df[llamaindex_metrics].notna().all(axis=1)]
        
        if df_llama.empty:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        metric_names = ['Correctness', 'Relevancy', 'Faithfulness', 'Semantic Similarity']
        
        for idx, (metric, name) in enumerate(zip(llamaindex_metrics, metric_names)):
            ax = axes[idx]
            
            # Plot vs Top-k
            top_k_values = sorted(df_llama['config_top_k'].unique())
            temperatures = sorted(df_llama['config_temperature'].unique())
            
            # Dynamic colors
            base_colors = [self.palette['matlab_blue'], self.palette['matlab_orange'], 
                          self.palette['matlab_green'], self.palette['matlab_purple'],
                          self.palette['matlab_yellow'], self.palette['matlab_cyan']]
            
            for i, temp in enumerate(temperatures):
                means = []
                stds = []
                
                for k in top_k_values:
                    data = df_llama[(df_llama['config_top_k'] == k) & 
                                   (df_llama['config_temperature'] == temp)][metric].values
                    if len(data) > 0:
                        means.append(np.mean(data))
                        stds.append(np.std(data))
                    else:
                        means.append(0)
                        stds.append(0)
                
                ax.errorbar(top_k_values, means, yerr=stds,
                           label=f'T={temp}',
                           color=base_colors[i % len(base_colors)],
                           marker='o', markersize=6, linewidth=2, capsize=3)
            
            ax.set_xlabel('Top-k')
            ax.set_ylabel(f'{name} Score')
            ax.set_title(f'{name} vs Top-k', fontsize=12)
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend(loc='best', fontsize=8)
        
        plt.suptitle('LlamaIndex Metrics vs Parameters', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'llamaindex_11_metrics_vs_parameters.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_ragas_plots(self, df: pd.DataFrame):
        """Create plots for RAGAS metrics if available"""
        # Check if RAGAS metrics are available
        ragas_metrics = ['ragas_faithfulness', 'ragas_answer_relevancy', 
                        'ragas_context_recall', 'ragas_context_precision']
        if not any(col in df.columns and df[col].notna().any() for col in ragas_metrics):
            logger.info("No RAGAS metrics found, skipping RAGAS plots")
            return
        
        logger.info("Creating RAGAS plots...")
        
        # Plot 12: RAGAS metrics heatmap
        self.plot_ragas_heatmap(df)
        
        # Plot 13: RAGAS metrics vs retrieval performance
        self.plot_ragas_vs_retrieval(df)
    
    def plot_ragas_heatmap(self, df: pd.DataFrame):
        """Plot 12: RAGAS metrics heatmap by configuration"""
        # Filter to only rows with RAGAS metrics
        ragas_metrics = ['ragas_faithfulness', 'ragas_answer_relevancy', 
                        'ragas_context_recall', 'ragas_context_precision']
        df_ragas = df[df[ragas_metrics].notna().all(axis=1)]
        
        if df_ragas.empty:
            return
        
        # Aggregate by configuration
        config_cols = ['config_top_k', 'config_threshold', 'config_temperature', 'config_response_mode']
        agg_data = df_ragas.groupby(config_cols)[ragas_metrics].mean().reset_index()
        
        # Create configuration labels
        agg_data['config_label'] = agg_data.apply(
            lambda row: f"k={row['config_top_k']},t={row['config_threshold']:.1f},T={row['config_temperature']:.1f}",
            axis=1
        )
        
        # Calculate composite score
        agg_data['ragas_composite'] = agg_data[ragas_metrics].mean(axis=1)
        top_configs = agg_data.nlargest(15, 'ragas_composite')
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data for heatmap
        heatmap_data = top_configs[ragas_metrics].T
        heatmap_data.columns = top_configs['config_label']
        heatmap_data.index = ['Faithfulness', 'Answer Relevancy', 'Context Recall', 'Context Precision']
        
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
                    vmin=0, vmax=1, ax=ax,
                    cbar_kws={'label': 'RAGAS Score'},
                    linewidths=0.5, linecolor='black')
        
        ax.set_xlabel('Configuration')
        ax.set_ylabel('RAGAS Metric')
        ax.set_title('RAGAS Metrics by Configuration (Top 15)', fontsize=16, pad=15)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'ragas_12_metrics_heatmap.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_ragas_vs_retrieval(self, df: pd.DataFrame):
        """Plot 13: RAGAS metrics vs retrieval performance"""
        ragas_metrics = ['ragas_faithfulness', 'ragas_answer_relevancy', 
                        'ragas_context_recall', 'ragas_context_precision']
        df_ragas = df[df[ragas_metrics].notna().all(axis=1)]
        
        if df_ragas.empty:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        metric_names = ['RAGAS Faithfulness', 'RAGAS Answer Relevancy', 
                       'RAGAS Context Recall', 'RAGAS Context Precision']
        
        for idx, (metric, name) in enumerate(zip(ragas_metrics, metric_names)):
            ax = axes[idx]
            
            # Scatter plot vs MAP score
            scatter = ax.scatter(df_ragas['map_score'], df_ragas[metric], 
                               c=df_ragas['config_top_k'], 
                               s=50, alpha=0.6, cmap='viridis')
            
            # Add trend line
            z = np.polyfit(df_ragas['map_score'], df_ragas[metric], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(df_ragas['map_score'].min(), df_ragas['map_score'].max(), 100)
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
            
            # Calculate correlation
            corr = df_ragas['map_score'].corr(df_ragas[metric])
            
            ax.set_xlabel('MAP Score')
            ax.set_ylabel(name)
            ax.set_title(f'{name} vs MAP (r={corr:.3f})', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Add colorbar for first plot
            if idx == 0:
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Top-k', rotation=270, labelpad=15)
        
        plt.suptitle('RAGAS Metrics vs Retrieval Performance', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'ragas_13_metrics_vs_retrieval.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_best_parameter_plots(self, df: pd.DataFrame):
        """Create plots to identify best parameter configurations"""
        logger.info("Creating best parameter identification plots...")
        
        # Plot 14: Composite score analysis
        self.plot_composite_scores(df)
        
        # Plot 15: Multi-objective Pareto frontier
        self.plot_multi_objective_pareto(df)
    
    def plot_composite_scores(self, df: pd.DataFrame):
        """Plot 14: Composite score analysis"""
        # Define weight scenarios
        weights_scenarios = {
            'Balanced': {
                'f1_token_score': 0.25,
                'map_score': 0.25,
                'exact_match': 0.25,
                'rouge_l': 0.25
            },
            'Quality-Focused': {
                'f1_token_score': 0.4,
                'exact_match': 0.4,
                'map_score': 0.1,
                'rouge_l': 0.1
            },
            'Retrieval-Focused': {
                'map_score': 0.4,
                'mrr_score': 0.3,
                'f1_token_score': 0.15,
                'exact_match': 0.15
            }
        }
        
        # Add LlamaIndex and RAGAS weights if available
        if 'correctness' in df.columns and df['correctness'].notna().any():
            weights_scenarios['Balanced']['correctness'] = 0.2
            weights_scenarios['Balanced']['f1_token_score'] = 0.2
            weights_scenarios['Balanced']['map_score'] = 0.2
            weights_scenarios['Balanced']['exact_match'] = 0.2
            weights_scenarios['Balanced']['rouge_l'] = 0.2
        
        if 'ragas_faithfulness' in df.columns and df['ragas_faithfulness'].notna().any():
            weights_scenarios['RAGAS-Focused'] = {
                'ragas_faithfulness': 0.3,
                'ragas_answer_relevancy': 0.3,
                'f1_token_score': 0.2,
                'map_score': 0.2
            }
        
        # Aggregate by configuration
        config_cols = ['config_top_k', 'config_threshold', 'config_temperature', 'config_response_mode']
        numeric_cols = [col for col in df.columns if col not in config_cols + ['query_id']]
        agg_df = df.groupby(config_cols)[numeric_cols].mean().reset_index()
        
        # Calculate composite scores
        for scenario_name, weights in weights_scenarios.items():
            # Filter weights to only include available metrics
            available_weights = {k: v for k, v in weights.items() if k in agg_df.columns and agg_df[k].notna().any()}
            if available_weights:
                # Normalize weights
                total_weight = sum(available_weights.values())
                normalized_weights = {k: v/total_weight for k, v in available_weights.items()}
                
                agg_df[f'composite_{scenario_name}'] = sum(
                    agg_df[metric] * weight for metric, weight in normalized_weights.items()
                )
        
        # Create plot
        n_scenarios = len(weights_scenarios)
        fig, axes = plt.subplots(1, n_scenarios, figsize=(6*n_scenarios, 6))
        if n_scenarios == 1:
            axes = [axes]
        
        for idx, (scenario_name, weights) in enumerate(weights_scenarios.items()):
            if f'composite_{scenario_name}' not in agg_df.columns:
                continue
                
            ax = axes[idx]
            
            # Get top 10 configurations
            top_configs = agg_df.nlargest(10, f'composite_{scenario_name}')
            
            # Create labels
            labels = []
            for _, row in top_configs.iterrows():
                label = f"k={row['config_top_k']},t={row['config_threshold']:.1f},T={row['config_temperature']:.1f}"
                labels.append(label)
            
            # Create bar plot
            y_pos = np.arange(len(labels))
            scores = top_configs[f'composite_{scenario_name}'].values
            
            bars = ax.barh(y_pos, scores, color=self.palette['matlab_blue'], alpha=0.8)
            
            # Add value labels
            for i, (bar, score) in enumerate(zip(bars, scores)):
                ax.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{score:.3f}', va='center', fontsize=9)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)
            ax.set_xlabel('Composite Score')
            ax.set_title(f'{scenario_name} Weighting', fontsize=14)
            ax.set_xlim(0, 1)
            ax.grid(True, axis='x', alpha=0.3)
        
        plt.suptitle('Top Configurations by Composite Score', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'best_params_14_composite_scores.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_multi_objective_pareto(self, df: pd.DataFrame):
        """Plot 15: Multi-objective Pareto frontier"""
        # Aggregate by configuration
        config_cols = ['config_top_k', 'config_threshold', 'config_temperature', 'config_response_mode']
        
        # Select metrics for Pareto analysis
        metrics = ['f1_token_score', 'map_score', 'total_time']
        if 'correctness' in df.columns and df['correctness'].notna().any():
            metrics.append('correctness')
        
        agg_df = df.groupby(config_cols)[metrics].mean().reset_index()
        
        # 2D Pareto frontier (F1 vs Latency)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Find Pareto frontier for F1 vs Latency
        pareto_indices = []
        for i, row in agg_df.iterrows():
            is_pareto = True
            for j, other in agg_df.iterrows():
                if i != j:
                    if (other['f1_token_score'] >= row['f1_token_score'] and
                        other['total_time'] <= row['total_time'] and
                        (other['f1_token_score'] > row['f1_token_score'] or
                         other['total_time'] < row['total_time'])):
                        is_pareto = False
                        break
            if is_pareto:
                pareto_indices.append(i)
        
        # Plot all points
        scatter1 = ax1.scatter(agg_df['total_time'], agg_df['f1_token_score'], 
                             c=agg_df['config_top_k'], s=50, alpha=0.6, cmap='viridis')
        
        # Highlight Pareto frontier
        pareto_df = agg_df.iloc[pareto_indices]
        ax1.scatter(pareto_df['total_time'], pareto_df['f1_token_score'],
                   c='red', s=200, marker='*', edgecolors='black', linewidth=2,
                   label='Pareto Optimal')
        
        # Connect Pareto points
        pareto_sorted = pareto_df.sort_values('total_time')
        ax1.plot(pareto_sorted['total_time'], pareto_sorted['f1_token_score'], 
                'r--', alpha=0.5, linewidth=2)
        
        ax1.set_xlabel('Total Time (s)')
        ax1.set_ylabel('F1 Score')
        ax1.set_title('Pareto Frontier: F1 Score vs Latency', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add colorbar
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('Top-k', rotation=270, labelpad=15)
        
        # MAP vs F1 tradeoff
        scatter2 = ax2.scatter(agg_df['map_score'], agg_df['f1_token_score'], 
                             c=agg_df['config_temperature'], s=50, alpha=0.6, cmap='coolwarm')
        
        ax2.set_xlabel('MAP Score')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('Retrieval vs Generation Quality Tradeoff', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('Temperature', rotation=270, labelpad=15)
        
        plt.suptitle('Multi-Objective Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'best_params_15_pareto_frontier.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_enhanced_summary_table(self, df: pd.DataFrame):
        """Create enhanced summary table with all metrics including LlamaIndex and RAGAS"""
        # Aggregate by configuration
        config_cols = ['config_top_k', 'config_threshold', 'config_temperature', 'config_response_mode']
        
        # Determine which metrics are available
        base_metrics = {
            'recall_at_k': 'mean',
            'precision_at_k': 'mean',
            'map_score': 'mean',
            'mrr_score': 'mean',
            'exact_match': 'mean',
            'f1_token_score': 'mean',
            'rouge_l': 'mean',
            'bleu_score': 'mean',
            'total_time': 'mean'
        }
        
        # Add LlamaIndex metrics if available
        llamaindex_metrics = ['correctness', 'relevancy', 'faithfulness', 'semantic_similarity']
        for metric in llamaindex_metrics:
            if metric in df.columns and df[metric].notna().any():
                base_metrics[metric] = 'mean'
        
        # Add RAGAS metrics if available
        ragas_metrics = ['ragas_faithfulness', 'ragas_answer_relevancy', 
                        'ragas_context_recall', 'ragas_context_precision']
        for metric in ragas_metrics:
            if metric in df.columns and df[metric].notna().any():
                base_metrics[metric] = 'mean'
        
        summary_df = df.groupby(config_cols).agg(base_metrics).reset_index()
        
        # Calculate composite score based on available metrics
        composite_weights = {
            'f1_token_score': 0.3,
            'map_score': 0.3,
            'exact_match': 0.2,
            'rouge_l': 0.2
        }
        
        # Add LlamaIndex to composite if available
        if 'correctness' in summary_df.columns:
            composite_weights['correctness'] = 0.2
            # Rebalance weights
            composite_weights = {k: v/1.2 for k, v in composite_weights.items()}
        
        # Add RAGAS to composite if available
        if 'ragas_faithfulness' in summary_df.columns:
            composite_weights['ragas_faithfulness'] = 0.15
            composite_weights['ragas_answer_relevancy'] = 0.15
            # Rebalance weights
            total = sum(composite_weights.values())
            composite_weights = {k: v/total for k, v in composite_weights.items()}
        
        # Calculate composite score
        summary_df['composite_score'] = sum(
            summary_df[metric] * weight 
            for metric, weight in composite_weights.items() 
            if metric in summary_df.columns
        )
        
        # Sort by composite score
        summary_df = summary_df.sort_values('composite_score', ascending=False).head(20)
        
        # Save to CSV
        summary_df.to_csv(self.plots_dir / 'enhanced_summary_table.csv', index=False)
        logger.info(f"Saved enhanced summary table to {self.plots_dir / 'enhanced_summary_table.csv'}")
        
        # Print best configuration
        if len(summary_df) > 0:
            best_config = summary_df.iloc[0]
            print("\n🏆 Best Configuration:")
            print(f"  Top-k: {best_config['config_top_k']}")
            print(f"  Threshold: {best_config['config_threshold']}")
            print(f"  Temperature: {best_config['config_temperature']}")
            print(f"  Response Mode: {best_config['config_response_mode']}")
            print(f"  Composite Score: {best_config['composite_score']:.3f}")
            
            # Print key metrics
            print("\nKey Metrics:")
            print(f"  F1 Score: {best_config['f1_token_score']:.3f}")
            print(f"  MAP Score: {best_config['map_score']:.3f}")
            print(f"  Exact Match: {best_config['exact_match']:.3f}")
            if 'correctness' in best_config:
                print(f"  LlamaIndex Correctness: {best_config['correctness']:.3f}")
            if 'ragas_faithfulness' in best_config:
                print(f"  RAGAS Faithfulness: {best_config['ragas_faithfulness']:.3f}")
        
        return summary_df
    
    def create_all_visualizations(self, detailed_results_file: str, summary_file: str):
        """Create all visualizations from evaluation results"""
        
        # Load data
        logger.info(f"Loading results from {detailed_results_file}")
        detailed_results = self.load_evaluation_results(detailed_results_file)
        summary = self.load_evaluation_results(summary_file)
        
        # Prepare DataFrame with confidence intervals
        logger.info("Preparing dataframe with normalized metrics...")
        df = self.prepare_dataframe_with_ci(detailed_results)
        
        # Create all plots
        logger.info("Creating retrieval plots...")
        self.create_retrieval_plots(df)
        
        logger.info("Creating generation plots...")
        self.create_generation_plots(df)
        
        logger.info("Creating bridging plot...")
        self.create_bridging_plot(df)
        
        # Create enhanced plots if metrics are available
        self.create_llamaindex_plots(df)
        self.create_ragas_plots(df)
        self.create_best_parameter_plots(df)
        
        logger.info("Creating tables...")
        self.create_tables(df)
        
        # Create enhanced summary table
        logger.info("Creating enhanced summary table...")
        self.create_enhanced_summary_table(df)
        
        logger.info(f"All visualizations saved to {self.plots_dir}")


def main():
    """Main function to create visualizations"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualization_and_reporting.py <results_timestamp>")
        print("Example: python visualization_and_reporting.py 20240101_120000")
        return
    
    timestamp = sys.argv[1]
    
    # Initialize visualizer
    visualizer = RefinedEvaluationVisualizer()
    
    # File paths
    detailed_file = f"../results/parameter_sweep_detailed_{timestamp}.json"
    summary_file = f"../results/parameter_sweep_summary_{timestamp}.json"
    
    # Check if files exist
    from pathlib import Path
    if not all(Path(f).exists() for f in [detailed_file, summary_file]):
        print(f"Error: Could not find all required files for timestamp {timestamp}")
        print("Required files:")
        print(f"  - {detailed_file}")
        print(f"  - {summary_file}")
        return
    
    # Create visualizations
    visualizer.create_all_visualizations(detailed_file, summary_file)
    
    print("\n✓ Refined visualization and reporting complete!")
    print(f"✓ Results saved to {visualizer.plots_dir}")
    print("\n📊 Generated plots:")
    print("   - 4 retrieval plots (recall, precision, MAP heatmap, MRR heatmap)")
    print("   - 4 generation plots (temperature effects, top-k effects, cost-quality scatter, per-query distributions)")  
    print("   - 1 bridging plot (correlation heatmap)")
    print("   - 4 summary tables")
    print("\n📊 Additional plots (if metrics available):")
    print("   - 2 LlamaIndex plots (metrics heatmap, metrics vs parameters)")
    print("   - 2 RAGAS plots (metrics heatmap, metrics vs retrieval)")
    print("   - 2 best parameter plots (composite scores, Pareto frontier)")
    print("   - 1 enhanced summary table (CSV with all metrics)")


if __name__ == "__main__":
    main()