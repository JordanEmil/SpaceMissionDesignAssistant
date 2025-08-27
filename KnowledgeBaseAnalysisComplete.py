import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
from tqdm import tqdm
from pathlib import Path

# --- Configuration --- PLEASE DELETE KNOWLEDGEBASEPLOTS FOLDER IF YOU WANT TO REGENERATE PLOTS, 
# and change paths for fast or slow methods
DATA_DIR = Path("data/Fast_Method_eoportal")
MANIFEST_PATH = DATA_DIR / "manifest_fast.jsonl"
TEXT_DATA_DIR = DATA_DIR / "text" / "missions"
PLOT_OUTPUT_DIR = Path("./Fast_KnowledgeBasePlots")
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

def load_manifest_data(manifest_path):
    """Load and parse manifest data, returning a DataFrame."""
    data = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['fetched_at'])
    df = df.sort_values('timestamp')
    return df

# --- Helper Functions ---

def extract_launch_year(text):
    """Extract launch year from mission text."""
    # Look for patterns like "launched in YYYY" or "launch date: YYYY"
    patterns = [
        r'launched?\s+(?:in\s+)?(\d{4})',
        r'launch\s+date[:\s]+.*?(\d{4})',
        r'(\d{4})\s+launch',
        r'launched\s+on\s+\d+\s+\w+\s+(\d{4})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            year = int(match.group(1))
            # Validate year is reasonable for space missions
            if 1957 <= year <= 2030:
                return year
    return None

def extract_agencies(text):
    """Extract space agency names from mission text."""
    # Common space agencies
    agency_patterns = {
        'NASA': r'\bNASA\b',
        'ESA': r'\bESA\b',
        'JAXA': r'\bJAXA\b',
        'Roscosmos': r'\bRoscosmos\b',
        'ISRO': r'\bISRO\b',
        'CNSA': r'\bCNSA\b',
        'CNES': r'\bCNES\b',
        'DLR': r'\bDLR\b',
        'ASI': r'\bASI\b',
        'CSA': r'\bCSA\b',
        'UKSA': r'\bUKSA\b',
        'KARI': r'\bKARI\b',
        'CONAE': r'\bCONAE\b',
        'NSPO': r'\bNSPO\b',
        'CAST': r'\bCAST\b',
        'Rosaviacosmos': r'\bRosaviacosmos\b',
        'NOAA': r'\bNOAA\b',
        'USGS': r'\bUSGS\b',
        'SpaceX': r'\bSpaceX\b',
        'Blue Origin': r'\bBlue Origin\b'
    }
    
    found_agencies = []
    for agency, pattern in agency_patterns.items():
        if re.search(pattern, text):
            found_agencies.append(agency)
    
    return found_agencies

# --- Scraper Performance Plots ---

def plot_scraping_timeline(df):
    """Plot scraping timeline and rate."""
    print("Generating Plot 1: Scraping Timeline...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    df['cumulative'] = range(1, len(df) + 1)
    ax1.plot(df['timestamp'], df['cumulative'], color=palette['matlab_blue'], linewidth=2.5)
    ax1.fill_between(df['timestamp'], 0, df['cumulative'], alpha=0.2, color=palette['matlab_blue'])
    ax1.set_ylabel('Cumulative Missions Downloaded')
    ax1.set_title('Scraping Progress Over Time', fontsize=18, weight='bold')
    
    window_size = 50
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
    df['rate'] = window_size / df['time_diff'].rolling(window=window_size).sum() * 60
    ax2.plot(df['timestamp'].iloc[window_size:], df['rate'].iloc[window_size:], 
             color=palette['matlab_orange'], linewidth=2)
    ax2.set_xlabel('Time (UTC)')
    ax2.set_ylabel('Missions per Minute')
    ax2.set_title(f'Scraping Rate', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_DIR / '1_scraping_timeline.pdf', dpi=300)
    plt.close()


def create_summary_statistics_table(df):
    """Create a summary statistics table visualization."""
    print("Generating Plot 2: Summary Statistics Table...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    total_missions = len(df)
    successful_missions = df['success'].sum()
    summary_data = [
        ['Metric', 'Value'],
        ['Total Missions Scraped', f'{total_missions:,}'],
        ['Successful Downloads (First Shot)', f'{successful_missions:,} ({successful_missions/total_missions*100:.1f}%)'],
        ['Total Tables Extracted', f"{df['num_tables'].sum():,}"],
        ['Total Images Catalogued', f"{df['num_images'].sum():,}"],
        ['Total Text Characters', f"{df['text_length'].sum():,}"],
        ['Total Data Size (GB)', f"{df['bytes'].sum() / (1024**3):.2f}"],
        ['Avg. Tables per Mission', f"{df['num_tables'].mean():.1f}"],
        ['Avg. Text Length (chars)', f"{df['text_length'].mean():,.0f}"],
    ]
    
    table = ax.table(cellText=summary_data, cellLoc='left', loc='center', colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 2.5)
    
    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor(palette['matlab_blue'])
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Scraping Summary Statistics', fontsize=18, weight='bold', pad=20)
    plt.savefig(PLOT_OUTPUT_DIR / '2_summary_statistics.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# --- Knowledge Base Analysis Plots ---

def plot_mission_launch_timeline(df_content):
    print("Generating Plot 3: Mission Launch Timeline...")
    df_content = df_content.dropna(subset=['launch_year']).copy()
    df_content['decade'] = (df_content['launch_year'] // 10 * 10).astype(int)
    decade_counts = df_content['decade'].value_counts().sort_index()
    
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(x=decade_counts.index, y=decade_counts.values, color=palette['matlab_blue'])
    ax.set_title('Mission Launches per Decade in the Knowledge Base', fontsize=18, weight='bold')
    ax.set_xlabel('Decade of Launch')
    ax.set_ylabel('Number of Missions')
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_DIR / '3_mission_launch_timeline.pdf', dpi=300)
    plt.close()

def plot_agency_distribution(df_content):
    print("Generating Plot 4: Agency Distribution...")
    all_agencies = [agency for sublist in df_content['agencies'] for agency in sublist]
    agency_counts = Counter(all_agencies)
    top_agencies = agency_counts.most_common(10)
    agency_df = pd.DataFrame(top_agencies, columns=['Agency', 'Count'])
    
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='Count', y='Agency', data=agency_df, color=palette['matlab_blue'])
    ax.set_title('Top 10 Mentioned Space Agencies/Companies Across Missions', fontsize=18, weight='bold')
    ax.set_xlabel('Number of Missions')
    ax.set_ylabel('')
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + 5, p.get_y() + p.get_height() / 2, f'{int(width)}', va='center')
    ax.set_xlim(0, agency_df['Count'].max() * 1.1)
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_DIR / '4_agency_distribution.pdf', dpi=300)
    plt.close()
    
def plot_mission_title_analysis(df):
    print("Generating Plot 5: Mission Title Analysis...")
    all_words = []
    stop_words = {'the', 'of', 'and', 'to', 'in', 'for', 'on', 'at', 'by', 'with', 'from', 'eoportal', '-'}
    for title in df['title'].dropna():
        words = re.findall(r'\b[A-Za-z-]+\b', title.lower())
        meaningful_words = [w for w in words if len(w) > 3 and w not in stop_words]
        all_words.extend(meaningful_words)
    word_freq = Counter(all_words)
    top_words = word_freq.most_common(20)
    word_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
    
    plt.figure(figsize=(12, 10))
    ax = sns.barplot(x='Frequency', y='Word', data=word_df, color=palette['matlab_blue'])
    ax.set_title('Top 20 Most Common Words in Mission Titles', fontsize=18, weight='bold')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('')
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_DIR / '5_mission_title_analysis.pdf', dpi=300)
    plt.close()

def plot_data_extraction_stats(df):
    print("Generating Plot 6: Data Extraction Stats...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Distribution of Extracted Content per Mission', fontsize=20, weight='bold')
    
    # Tables
    ax1.hist(df['num_tables'], bins=range(0, df['num_tables'].max() + 2), color=palette['matlab_blue'], align='left', rwidth=0.8)
    ax1.set_title('Tables per Mission')
    ax1.set_xlabel('Number of Tables')
    ax1.set_ylabel('Number of Missions')

    # Images
    ax2.hist(df['num_images'], bins=30, color=palette['matlab_orange'])
    ax2.set_title('Images per Mission')
    ax2.set_xlabel('Number of Images')
    
    # Text length
    ax3.hist(df['text_length'] / 1000, bins=50, color=palette['matlab_yellow'])
    ax3.set_title('Extracted Text Length')
    ax3.set_xlabel('Text Length (thousands of characters)')
    ax3.set_ylabel('Number of Missions')

    # File size
    ax4.hist(df['bytes'] / 1024, bins=50, color=palette['matlab_purple'])
    ax4.set_title('HTML File Size')
    ax4.set_xlabel('HTML File Size (KB)')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(PLOT_OUTPUT_DIR / '6_data_extraction_stats.pdf', dpi=300)
    plt.close()






def main():
    setup_plot_theme()
    
    if not MANIFEST_PATH.exists():
        print(f"Error: Manifest file not found at '{MANIFEST_PATH}'")
        return

    # --- Part A: Scraper Performance ---
    print("--- Section A: Visualizing Scraper Performance ---")
    df_manifest = load_manifest_data(MANIFEST_PATH)
    plot_scraping_timeline(df_manifest)
    create_summary_statistics_table(df_manifest)
    
    # --- Part B: Knowledge Base Analysis ---
    print("\n--- Section B: Analyzing Knowledge Base Content ---")
    content_data = []
    if not TEXT_DATA_DIR.exists():
        print(f"Warning: Text data directory not found at '{TEXT_DATA_DIR}'. Skipping content-based plots.")
    else:
        for record in tqdm(df_manifest.to_dict('records'), desc="Reading text files for content analysis"):
            if record['success']:
                filepath = Path(record['local_text'])
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                        content_data.append({
                            'filename': filepath.name,
                            'launch_year': extract_launch_year(text),
                            'agencies': extract_agencies(text),
                        })
                except Exception:
                    continue
    df_content = pd.DataFrame(content_data)

    plot_mission_launch_timeline(df_content)
    plot_agency_distribution(df_content)
    plot_mission_title_analysis(df_manifest) # Titles are in the manifest
    plot_data_extraction_stats(df_manifest) # Extraction stats are in the manifest
    

if __name__ == "__main__":
    main()