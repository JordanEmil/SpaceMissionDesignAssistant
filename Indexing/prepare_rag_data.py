#!/usr/bin/env python3
"""
Prepare eoPortal data for LlamaIndex RAG pipeline
Organizes and structures the scraped data for optimal chunking and embedding
"""

import json
import re
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timezone
import pandas as pd

MIN_TEXT_CHARS = 800

def normalize_text(s: str) -> str:
    s = s.replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n[ \t]+\n", "\n\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

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

def prepare_rag_directory_structure(base_dir: Path) -> Dict[str, Path]:
    """Create organized directory structure for RAG pipeline"""
    rag_dir = base_dir / "rag_ready_data"
    
    # Create main directories
    dirs = {
        'metadata': rag_dir / 'metadata',
        'combined': rag_dir / 'combined_documents'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs

def create_mission_metadata(manifest_path: Path, output_dir: Path) -> pd.DataFrame:
    """Create comprehensive metadata for each mission"""
    metadata_list = []
    
    with open(manifest_path, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                metadata = {
                    'mission_id': entry['mission_name'],
                    'title': entry['title'],
                    'url': entry['url'],
                    'text_length': entry['text_length'],
                    'num_tables': entry['num_tables'],
                    'num_images': entry['num_images'],
                    'file_size_kb': entry['bytes'] / 1024,
                    'scraped_date': entry['fetched_at'],
                    'success': entry['success'],
                    'sha256': entry.get('sha256', ''),
                    'source': entry.get('source', 'eoportal')
                }
                metadata_list.append(metadata)
            except json.JSONDecodeError:
                continue
    
    df = pd.DataFrame(metadata_list)
    
    # # Filter to only successful missions, not needed as we manually retry failures
    # df = df[df["success"]].copy()
    
    df.to_csv(output_dir / 'mission_metadata.csv', index=False)
    
    # Write prep run info
    (output_dir / 'prep_run.json').write_text(
        json.dumps({
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "min_text_chars": MIN_TEXT_CHARS,
            "notes": "eoPortal prep for RAG"
        }, indent=2),
        encoding='utf-8'
    )
    
    return df

def combine_mission_data(source_dir: Path, output_dir: Path, metadata_df: pd.DataFrame):
    """Combine text, tables, and metadata into unified documents for each mission"""
    
    for _, mission in metadata_df.iterrows():            
        mission_id = mission['mission_id']
        
        # Load text content
        text_file = source_dir / 'text' / 'missions' / f'{mission_id}.txt'
        if not text_file.exists():
            continue
            
        text = normalize_text(text_file.read_text(encoding='utf-8', errors='ignore'))
        if len(text) < MIN_TEXT_CHARS:
            continue
        
        # Load tables and images
        tables_file = source_dir / 'structured' / 'missions' / f'{mission_id}_tables.json'
        images_file = source_dir / 'structured' / 'missions' / f'{mission_id}_images.json'
        tables = json.load(tables_file.open()) if tables_file.exists() else []
        images = json.load(images_file.open()) if images_file.exists() else []
        
        has_tables = bool(tables)
        has_images = bool(images)
        
        # Create retrieval header
        header_lines = []
        if mission['title']: 
            header_lines.append(f"# {mission['title']}")
        if mission_id:
            header_lines.append(f"Mission: {mission_id}")
        if mission['url']:
            header_lines.append(f"Source: {mission['url']}")
        header = "\n".join(header_lines)
        
        # Optional: append tables as markdown blocks
        tables_md_blocks = []
        for t in tables:
            rows = t.get("rows", [])
            md = markdown_table(rows)
            if md:
                tables_md_blocks.append("\n**Table (eoPortal)**\n" + md)
        
        combined_text = "\n\n".join([p for p in [header, text, "\n".join(tables_md_blocks)] if p])
        
        combined_doc = {
            'mission_id': mission_id,
            'title': mission['title'],
            'url': mission['url'],
            'has_tables': has_tables,
            'has_images': has_images,
            'metadata': mission.to_dict(),
            'text': combined_text,
            'tables': tables,
            'images': images
        }
        
        # Save combined document
        output_file = output_dir / f'{mission_id}_combined.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(combined_doc, f, indent=2, ensure_ascii=False)

def create_document_index(combined_dir: Path, output_file: Path):
    """Create an index of all documents with key metadata for quick access"""
    index = []
    
    for doc_file in combined_dir.glob('*_combined.json'):
        with open(doc_file, 'r', encoding='utf-8') as f:
            doc = json.load(f)
            
        index_entry = {
            'mission_id': doc['mission_id'],
            'title': doc['title'],
            'url': doc['url'],
            'file_path': str(doc_file),
            'text_length': len(doc.get('text', '')),
            'num_tables': len(doc.get('tables', [])),
            'num_images': len(doc.get('images', [])),
            'has_tables': doc.get('has_tables', False),
            'has_images': doc.get('has_images', False)
        }
        index.append(index_entry)
    
    # Save index
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2)
    
    return index


def generate_rag_statistics(dirs: Dict[str, Path], metadata_df: pd.DataFrame):
    """Generate statistics about the RAG-ready data"""
    stats = {
        'total_missions': int(len(metadata_df)),
        'successful_missions': int(len(metadata_df[metadata_df['success']])),
        'total_text_chars': int(metadata_df['text_length'].sum()),
        'avg_text_per_mission': float(metadata_df['text_length'].mean()),
        'total_tables': int(metadata_df['num_tables'].sum()),
        'total_images': int(metadata_df['num_images'].sum()),
        'storage_size_mb': float(metadata_df['file_size_kb'].sum() / 1024)
    }
    
    with open(dirs['metadata'] / 'rag_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats

def main():
    """Main function to prepare data for RAG pipeline"""
    # Determine if we're running from the Indexing directory or parent directory
    current_dir = Path.cwd()
    if current_dir.name == "Indexing":
        # Running from within Indexing directory
        source_dir = Path("../data/Fast_Method_eoportal")
        base_dir = Path("..")
    else:
        # Running from parent directory (SpaceMissionDesignAssistant)
        source_dir = Path("data/Fast_Method_eoportal")
        base_dir = Path(".")
    
    print("Preparing data for LlamaIndex RAG pipeline...")
    
    # Create directory structure
    dirs = prepare_rag_directory_structure(base_dir)
    print(f"✓ Created RAG directory structure at {base_dir / 'rag_ready_data'}")
    
    # Create metadata
    print("\nCreating mission metadata...")
    manifest_path = source_dir / "manifest_fast.jsonl"
    metadata_df = create_mission_metadata(manifest_path, dirs['metadata'])
    print(f"✓ Created metadata for {len(metadata_df)} missions")
    
    # Combine mission data
    print("\nCombining mission data...")
    combine_mission_data(source_dir, dirs['combined'], metadata_df)
    print(f"✓ Combined text, tables, and images for each mission")
    
    # Create document index
    print("\nCreating document index...")
    index = create_document_index(dirs['combined'], dirs['metadata'] / 'document_index.json')
    print(f"✓ Indexed {len(index)} documents")
    
    
    # Generate statistics
    print("\nGenerating RAG statistics...")
    stats = generate_rag_statistics(dirs, metadata_df)
    
    print("\n" + "="*50)
    print("RAG Data Preparation Complete!")
    print("="*50)
    print(f"Total missions: {stats['total_missions']:,}")
    print(f"Total text: {stats['total_text_chars']:,} characters")
    print(f"Total tables: {stats['total_tables']:,}")
    print(f"Total images: {stats['total_images']:,}")
    print(f"Storage size: {stats['storage_size_mb']:.1f} MB")
    
    print("\nData is now ready for RAG pipeline:")
    print("- Combined documents available in 'rag_ready_data/combined_documents/'")
    print("- Metadata available in 'rag_ready_data/metadata/'")
    print("- Document index available for quick access")

if __name__ == "__main__":
    main()