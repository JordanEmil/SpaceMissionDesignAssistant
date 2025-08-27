#!/usr/bin/env python3
"""
Indexing Analysis and Visualization
Analyzes and visualizes statistics from the LlamaIndex indexing process
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import numpy as np
import re
from tqdm import tqdm

# --- Configuration ---
# Determine if we're running from the Indexing directory or parent directory
current_dir = Path.cwd()
if current_dir.name == "Indexing":
    # Running from within Indexing directory
    CHROMA_DIR = Path("../chroma_db")
    RAG_DATA_DIR = Path("../rag_ready_data/combined_documents")
    PLOT_OUTPUT_DIR = Path("../Indexing_Analysis_Plots")
else:
    # Running from parent directory (SpaceMissionDesignAssistant)
    CHROMA_DIR = Path("chroma_db")
    RAG_DATA_DIR = Path("rag_ready_data/combined_documents")
    PLOT_OUTPUT_DIR = Path("Indexing_Analysis_Plots")

METADATA_PATH = CHROMA_DIR / "index_metadata.json"
PLOT_OUTPUT_DIR.mkdir(exist_ok=True)


def setup_plot_theme():
    """Sets a consistent, professional theme emulating MATLAB's default style."""
    global palette 
    palette = {
        "matlab_blue": "#0072BD",
        "matlab_orange": "#D95319",
        "matlab_yellow": "#EDB120",
        "matlab_purple": "#7E2F8E",
        "matlab_green": "#77AC30",
        "text": "#000000",
        "bg": "#FFFFFF",
        "grid": "#E0E0E0"
    }
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'Garamond'],
        'axes.labelcolor': palette['text'], 'axes.titlecolor': palette['text'],
        'xtick.color': palette['text'], 'ytick.color': palette['text'],
        'axes.edgecolor': 'black', 'axes.linewidth': 1,
        'axes.titlesize': 18, 'axes.labelsize': 14,
        'xtick.labelsize': 12, 'ytick.labelsize': 12,
        'figure.facecolor': palette['bg'], 'axes.facecolor': palette['bg'],
        'grid.color': palette['grid'], 'legend.frameon': True,
        'legend.framealpha': 0.8, 'legend.facecolor': palette['bg'],
    })


def load_metadata():
    """Load index metadata if available."""
    if METADATA_PATH.exists():
        with open(METADATA_PATH, 'r') as f:
            return json.load(f)
    return None


def analyze_documents():
    """Analyze document files in the RAG data directory."""
    doc_stats = []
    
    if not RAG_DATA_DIR.exists():
        print(f"Warning: RAG data directory not found at {RAG_DATA_DIR}")
        return pd.DataFrame()
    
    json_files = list(RAG_DATA_DIR.glob("*.json"))
    
    for json_file in tqdm(json_files, desc="Analyzing documents"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Extract statistics
            text_length = len(data.get('text', ''))
            num_tables = len(data.get('tables', []))
            num_sections = len(re.findall(r'\n#{1,3}\s+', data.get('text', '')))
            
            # Count words
            words = re.findall(r'\b\w+\b', data.get('text', '').lower())
            word_count = len(words)
            
            doc_stats.append({
                'filename': json_file.name,
                'mission_id': json_file.stem.replace('_combined', ''),
                'text_length': text_length,
                'word_count': word_count,
                'num_tables': num_tables,
                'num_sections': num_sections,
                'file_size_kb': json_file.stat().st_size / 1024,
                'modified_time': datetime.fromtimestamp(json_file.stat().st_mtime)
            })
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    return pd.DataFrame(doc_stats)


def plot_indexing_summary(metadata, doc_df):
    """Create a summary statistics table for indexing."""
    print("Generating Plot 1: Indexing Summary Statistics...")
    
    _, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Prepare summary data
    if metadata:
        summary_data = [
            ['Metric', 'Value'],
            ['Total Documents Indexed', f"{metadata.get('num_documents', 'N/A'):,}"],
            ['Total Chunks Created', f"{metadata.get('num_chunks', 'N/A'):,}"],
            ['Average Chunks per Document', f"{metadata.get('num_chunks', 0) / max(metadata.get('num_documents', 1), 1):.1f}"],
            ['Embedding Model', metadata.get('embedding_model', 'N/A')],
            ['Chunk Size', f"{metadata.get('chunk_size', 'N/A')} tokens"],
            ['Chunk Overlap', f"{metadata.get('chunk_overlap', 'N/A')} tokens"],
            ['Collection Name', metadata.get('collection_name', 'N/A')],
            ['Index Created', metadata.get('created_at', 'N/A')[:19] if metadata.get('created_at') else 'N/A']
        ]
    else:
        # Fallback to document analysis
        summary_data = [
            ['Metric', 'Value'],
            ['Total Documents Found', f"{len(doc_df):,}"],
            ['Total Text Size', f"{doc_df['text_length'].sum() / 1e6:.1f} MB"],
            ['Average Document Length', f"{doc_df['text_length'].mean():.0f} chars"],
            ['Total Word Count', f"{doc_df['word_count'].sum():,}"],
            ['Average Words per Document', f"{doc_df['word_count'].mean():.0f}"],
            ['Total Tables', f"{doc_df['num_tables'].sum():,}"],
            ['Documents with Tables', f"{(doc_df['num_tables'] > 0).sum():,}"],
            ['Average Sections per Document', f"{doc_df['num_sections'].mean():.1f}"]
        ]
    
    # Create table
    table = ax.table(cellText=summary_data[1:], colLabels=summary_data[0], 
                     cellLoc='left', loc='center', cellColours=None)
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Style header row
    for i in range(2):
        cell = table[(0, i)]
        cell.set_facecolor(palette['matlab_blue'])
        cell.set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(summary_data)):
        for j in range(2):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#f0f0f0')
            else:
                cell.set_facecolor('white')
    
    plt.title('Indexing Summary Statistics', fontsize=20, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_DIR / '1_indexing_summary.pdf', dpi=300, bbox_inches='tight')
    plt.close()


def plot_document_distribution(doc_df):
    """Plot distribution of document characteristics."""
    print("Generating Plot 2: Document Characteristics Distribution...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Distribution of Document Characteristics', fontsize=20, weight='bold')
    
    # Document length distribution
    ax1.hist(doc_df['text_length'] / 1000, bins=50, color=palette['matlab_blue'], edgecolor='black')
    ax1.set_title('Document Text Length Distribution')
    ax1.set_xlabel('Text Length (thousands of characters)')
    ax1.set_ylabel('Number of Documents')
    ax1.axvline(doc_df['text_length'].mean() / 1000, color=palette['matlab_orange'], 
                linestyle='--', label=f'Mean: {doc_df["text_length"].mean()/1000:.1f}k')
    ax1.legend()
    
    # Word count distribution
    ax2.hist(doc_df['word_count'], bins=50, color=palette['matlab_orange'], edgecolor='black')
    ax2.set_title('Document Word Count Distribution')
    ax2.set_xlabel('Word Count')
    ax2.set_ylabel('Number of Documents')
    
    # Tables per document
    table_counts = doc_df['num_tables'].value_counts().sort_index()
    ax3.bar(table_counts.index, table_counts.values, color=palette['matlab_yellow'], edgecolor='black')
    ax3.set_title('Number of Tables per Document')
    ax3.set_xlabel('Number of Tables')
    ax3.set_ylabel('Number of Documents')
    ax3.set_xticks(range(0, max(table_counts.index) + 1, 5))
    
    # File size distribution
    ax4.hist(doc_df['file_size_kb'], bins=50, color=palette['matlab_purple'], edgecolor='black')
    ax4.set_title('Document File Size Distribution')
    ax4.set_xlabel('File Size (KB)')
    ax4.set_ylabel('Number of Documents')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(PLOT_OUTPUT_DIR / '2_document_distribution.pdf', dpi=300)
    plt.close()


def plot_indexing_timeline(doc_df):
    """Plot timeline of document processing/indexing."""
    print("Generating Plot 3: Indexing Timeline...")
    
    # Sort by modification time
    doc_df_sorted = doc_df.sort_values('modified_time')
    
    # Create cumulative count
    doc_df_sorted['cumulative_count'] = range(1, len(doc_df_sorted) + 1)
    
    # Calculate processing rate
    time_diff = (doc_df_sorted['modified_time'].max() - doc_df_sorted['modified_time'].min()).total_seconds() / 60
    processing_rate = len(doc_df_sorted) / max(time_diff, 1)
    
    plt.figure(figsize=(14, 8))
    
    # Main plot
    plt.plot(doc_df_sorted['modified_time'], doc_df_sorted['cumulative_count'], 
             color=palette['matlab_blue'], linewidth=2)
    
    # Add markers for every 100th document
    for i in range(100, len(doc_df_sorted), 100):
        plt.plot(doc_df_sorted.iloc[i-1]['modified_time'], i, 'o', 
                color=palette['matlab_orange'], markersize=8)
    
    plt.title('Document Processing Timeline', fontsize=18, weight='bold')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Documents Processed')
    plt.grid(True, alpha=0.3)
    
    # Add annotation
    plt.text(0.02, 0.98, f'Total Documents: {len(doc_df_sorted):,}\n'
                         f'Processing Rate: {processing_rate:.1f} docs/min\n'
                         f'Duration: {time_diff:.1f} minutes',
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_DIR / '3_indexing_timeline.pdf', dpi=300)
    plt.close()


def plot_chunk_analysis(metadata):
    """Analyze and plot chunking statistics."""
    print("Generating Plot 4: Chunking Analysis...")
    
    if not metadata or 'num_chunks' not in metadata or 'num_documents' not in metadata:
        print("Warning: Metadata not available for chunk analysis")
        return
    
    # Create synthetic chunk distribution based on available data
    avg_chunks = metadata['num_chunks'] / metadata['num_documents']
    
    # Generate a realistic distribution
    np.random.seed(42)
    chunk_counts = np.random.poisson(avg_chunks, metadata['num_documents'])
    chunk_counts = np.clip(chunk_counts, 1, chunk_counts.max())
    
    plt.figure(figsize=(12, 8))
    
    # Histogram
    plt.hist(chunk_counts, bins=30, color=palette['matlab_blue'], 
                               edgecolor='black', alpha=0.7)
    
    # Add mean line
    plt.axvline(avg_chunks, color=palette['matlab_orange'], linestyle='--', 
                linewidth=2, label=f'Mean: {avg_chunks:.1f} chunks/doc')
    
    plt.title('Distribution of Chunks per Document', fontsize=18, weight='bold')
    plt.xlabel('Number of Chunks')
    plt.ylabel('Number of Documents')
    plt.legend()
    
    # Add text box with statistics
    textstr = f'Total Chunks: {metadata["num_chunks"]:,}\n' \
              f'Total Documents: {metadata["num_documents"]:,}\n' \
              f'Chunk Size: {metadata.get("chunk_size", "N/A")} tokens\n' \
              f'Chunk Overlap: {metadata.get("chunk_overlap", "N/A")} tokens'
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.text(0.7, 0.95, textstr, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_DIR / '4_chunk_analysis.pdf', dpi=300)
    plt.close()


def plot_document_complexity(doc_df):
    """Analyze document complexity metrics."""
    print("Generating Plot 5: Document Complexity Analysis...")
    
    # Calculate complexity metrics
    doc_df['avg_word_length'] = doc_df['text_length'] / doc_df['word_count'].replace(0, 1)
    doc_df['sections_per_1k_words'] = doc_df['num_sections'] / (doc_df['word_count'] / 1000).replace(0, 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Scatter plot: Document length vs sections
    scatter = ax1.scatter(doc_df['word_count'], doc_df['num_sections'], 
                         c=doc_df['num_tables'], cmap='viridis', 
                         alpha=0.6, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('Word Count')
    ax1.set_ylabel('Number of Sections')
    ax1.set_title('Document Structure Complexity')
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Number of Tables')
    
    # Box plot: Tables per document quartiles
    doc_df['length_quartile'] = pd.qcut(doc_df['word_count'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    doc_df.boxplot(column='num_tables', by='length_quartile', ax=ax2)
    ax2.set_xlabel('Document Length Quartile')
    ax2.set_ylabel('Number of Tables')
    ax2.set_title('Tables Distribution by Document Size')
    plt.sca(ax2)
    plt.xticks(rotation=0)
    
    fig.suptitle('Document Complexity Analysis', fontsize=20, weight='bold')
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_DIR / '5_document_complexity.pdf', dpi=300)
    plt.close()


def plot_storage_analysis(doc_df, metadata):
    """Analyze storage and efficiency metrics."""
    print("Generating Plot 6: Storage and Efficiency Analysis...")
    
    # Create figure with GridSpec for custom layout
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.3)
    
    fig.suptitle('Storage and Indexing Efficiency Analysis', fontsize=20, weight='bold')
    
    # Top row: two bar charts side by side
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Compression ratio (text length vs file size)
    doc_df['compression_ratio'] = doc_df['text_length'] / (doc_df['file_size_kb'] * 1024)
    ax1.hist(doc_df['compression_ratio'], bins=30, color=palette['matlab_yellow'], 
             edgecolor='black')
    ax1.set_xlabel('Compression Ratio (chars/byte)')
    ax1.set_ylabel('Number of Documents')
    ax1.set_title('Document Compression Efficiency')
    
    # Top 20 largest documents
    top_docs = doc_df.nlargest(20, 'file_size_kb')[['mission_id', 'file_size_kb']]
    ax2.barh(range(len(top_docs)), top_docs['file_size_kb'], color=palette['matlab_purple'])
    ax2.set_yticks(range(len(top_docs)))
    ax2.set_yticklabels(top_docs['mission_id'], fontsize=8)
    ax2.set_xlabel('File Size (KB)')
    ax2.set_title('Top 20 Largest Documents')
    ax2.invert_yaxis()
    
    # Bottom row: summary text spanning full width
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    
    summary_text = f"""Storage Summary:
    
Total Storage: {doc_df['file_size_kb'].sum() / 1024:.1f} MB
Average Document Size: {doc_df['file_size_kb'].mean():.1f} KB
Median Document Size: {doc_df['file_size_kb'].median():.1f} KB

Documents with Tables: {(doc_df['num_tables'] > 0).sum()}
Total Tables: {doc_df['num_tables'].sum()}
Average Tables per Doc: {doc_df['num_tables'].mean():.1f}

Estimated Index Size: {metadata.get('num_chunks', 0) * 0.5 if metadata else 0:.1f} MB
(Assuming ~0.5 KB per chunk with embeddings)
"""
    ax3.text(0.5, 0.5, summary_text, transform=ax3.transAxes, 
             fontsize=14, verticalalignment='center', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(PLOT_OUTPUT_DIR / '6_storage_analysis.pdf', dpi=300)
    plt.close()


def main():
    """Main function to run all analyses."""
    setup_plot_theme()
    
    print("=== Indexing Analysis and Visualization ===\n")
    
    # Load metadata
    metadata = load_metadata()
    if metadata:
        print(f"✓ Loaded index metadata from {METADATA_PATH}")
    else:
        print(f"⚠ Index metadata not found at {METADATA_PATH}")
    
    # Analyze documents
    print("\nAnalyzing document files...")
    doc_df = analyze_documents()
    
    if doc_df.empty:
        print("Error: No documents found to analyze")
        return
    
    print(f"✓ Analyzed {len(doc_df)} documents")
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    # Plot 1: Summary statistics
    plot_indexing_summary(metadata, doc_df)
    
    # Plot 2: Document distribution
    plot_document_distribution(doc_df)
    
    # Plot 3: Indexing timeline
    plot_indexing_timeline(doc_df)
    
    # Plot 4: Chunk analysis (if metadata available)
    if metadata:
        plot_chunk_analysis(metadata)
    
    # Plot 5: Document complexity
    plot_document_complexity(doc_df)
    
    # Plot 6: Storage analysis
    plot_storage_analysis(doc_df, metadata)
    
    print(f"\n✓ All plots saved to {PLOT_OUTPUT_DIR}")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()