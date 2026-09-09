#!/usr/bin/env python3
"""
research: Download the leader's FineWeb simplification pairs dataset and verify.
This script requires network access to retrieve the dataset.

Strategy: Download go76dof/Fineweb_simplification_pairs from HuggingFace,
verify word count and format, then make it available for training.

If direct download fails, fall back to downloading FineWeb-Edu source text
and generating our own simplifications with Qwen3.5-9B.
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import os
import sys
import json
import pathlib
import hashlib
from datetime import datetime

# Determine workspace from environment or script location
WORKSPACE = os.environ.get("QIUSHI_AI_LAB_WORKSPACE", 
    str(_public_path('experiments/archive/representation_and_objectives')))
RUN_DIR = os.environ.get("QIUSHI_AI_LAB_RUN_DIR",
    str(pathlib.Path(WORKSPACE) / "training" / "runs" / "data_download"))
DATA_DIR = pathlib.Path(WORKSPACE) / "training" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = pathlib.Path(RUN_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def try_download_leader_dataset():
    """Try to download the leader's exact dataset from HuggingFace."""
    print("[1] Attempting to download go76dof/Fineweb_simplification_pairs...")
    
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        
        # List files in the dataset repo
        files = list_repo_files("go76dof/Fineweb_simplification_pairs", repo_type="dataset")
        print(f"  Found {len(files)} files: {files}")
        
        # Download the training file
        train_file = None
        for f in files:
            if 'train' in f.lower() or f.endswith('.txt') or f.endswith('.train'):
                train_file = f
                break
        
        if not train_file and files:
            train_file = files[0]  # Try first file
            
        if train_file:
            local_path = hf_hub_download(
                "go76dof/Fineweb_simplification_pairs",
                train_file,
                repo_type="dataset",
                local_dir=str(DATA_DIR / "fineweb_simplification_pairs")
            )
            print(f"  Downloaded: {local_path}")
            return local_path
        else:
            print("  No suitable file found in dataset repo")
            return None
            
    except Exception as e:
        print(f"  Failed: {e}")
        return None

def try_download_fineweb_edu_sample():
    """Download a sample of FineWeb-Edu for our own processing."""
    print("[2] Attempting to download FineWeb-Edu sample...")
    
    try:
        from datasets import load_dataset
        
        # Load a streaming sample of FineWeb-Edu
        # We need ~5M words of source text (since pairs will be ~10M total)
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            split="train",
            streaming=True,
            trust_remote_code=True
        )
        
        output_path = DATA_DIR / "fineweb_edu_sample.txt"
        word_count = 0
        target_words = 5_500_000  # ~5.5M words of source (simplified will add ~4.5M more)
        doc_count = 0
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in ds:
                text = example.get('text', '')
                if not text.strip():
                    continue
                    
                words = text.split()
                word_count += len(words)
                doc_count += 1
                f.write(text.strip() + '\n\n')
                
                if word_count >= target_words:
                    break
                    
                if doc_count % 1000 == 0:
                    print(f"  Progress: {doc_count} docs, {word_count:,} words")
        
        print(f"  Downloaded {doc_count} docs, {word_count:,} words to {output_path}")
        return str(output_path)
        
    except Exception as e:
        print(f"  Failed: {e}")
        return None

def verify_data(path):
    """Verify the downloaded data meets BabyLM constraints."""
    print(f"\n[3] Verifying data at {path}...")
    
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    words = text.split()
    lines = text.split('\n')
    non_empty_lines = [l for l in lines if l.strip()]
    
    stats = {
        "path": str(path),
        "file_size_bytes": os.path.getsize(path),
        "total_words": len(words),
        "total_lines": len(lines),
        "non_empty_lines": len(non_empty_lines),
        "verified_utc": datetime.utcnow().isoformat(),
        "within_10M_budget": len(words) <= 10_000_000,
    }
    
    print(f"  Words: {stats['total_words']:,}")
    print(f"  Lines: {stats['total_lines']:,}")
    print(f"  Non-empty lines: {stats['non_empty_lines']:,}")
    print(f"  Within 10M budget: {stats['within_10M_budget']}")
    
    # Check for pair structure (blank-line separated)
    pairs = text.split('\n\n')
    non_empty_pairs = [p for p in pairs if p.strip()]
    stats["pair_count"] = len(non_empty_pairs)
    print(f"  Pairs (blank-line separated blocks): {stats['pair_count']:,}")
    
    # Save verification
    verify_path = OUTPUT_DIR / "data_verification.json"
    with open(verify_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Verification saved: {verify_path}")
    
    return stats

def main():
    print(f"=== FineWeb Data Acquisition ===")
    print(f"Workspace: {WORKSPACE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Started: {datetime.utcnow().isoformat()}")
    print()
    
    # Strategy 1: Try leader's exact dataset
    result = try_download_leader_dataset()
    
    if result and os.path.exists(result):
        stats = verify_data(result)
        summary = {
            "status": "leader_dataset_downloaded",
            "source": "go76dof/Fineweb_simplification_pairs",
            "path": result,
            "stats": stats,
        }
    else:
        # Strategy 2: Download FineWeb-Edu sample for our own processing
        result = try_download_fineweb_edu_sample()
        
        if result and os.path.exists(result):
            stats = verify_data(result)
            summary = {
                "status": "fineweb_edu_sample_downloaded",
                "source": "HuggingFaceFW/fineweb-edu",
                "path": result,
                "stats": stats,
                "next_step": "Generate simplifications with Qwen3.5-9B",
            }
        else:
            summary = {
                "status": "download_failed",
                "error": "Neither leader dataset nor FineWeb-Edu could be downloaded",
                "next_step": "Generate factual content with Qwen3.5-9B from scratch",
            }
    
    # Save summary
    summary_path = OUTPUT_DIR / "download_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {summary_path}")
    return 0 if summary["status"] != "download_failed" else 1

if __name__ == "__main__":
    sys.exit(main())
