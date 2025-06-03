#!/usr/bin/env python3
"""
Script to pre-download Whisper models for local use.
Run this before building your Docker image or deploying.
"""

import os
import argparse
import whisper
from pathlib import Path

def download_model(model_size, model_dir):
    """Download and save Whisper model locally"""
    print(f"Downloading Whisper {model_size} model...")
    
    # Create model directory
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Download model
    model = whisper.load_model(model_size, download_root=str(model_dir))
    
    model_path = model_dir / f"{model_size}.pt"
    print(f"Model downloaded and saved to: {model_path}")
    
    # Verify file exists and get size
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"Model file size: {size_mb:.1f} MB")
        return True
    else:
        print("Error: Model file not found after download")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download Whisper models locally")
    parser.add_argument(
        "--model", 
        choices=["tiny", "base", "small", "medium", "large"],
        default="small",
        help="Model size to download (default: small)"
    )
    parser.add_argument(
        "--dir",
        default="./models",
        help="Directory to save models (default: ./models)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all model sizes"
    )
    
    args = parser.parse_args()
    
    if args.all:
        models = ["tiny", "base", "small", "medium", "large"]
        print("Downloading all Whisper models...")
        for model_size in models:
            success = download_model(model_size, args.dir)
            if not success:
                print(f"Failed to download {model_size} model")
                return 1
        print("All models downloaded successfully!")
    else:
        success = download_model(args.model, args.dir)
        if not success:
            return 1
    
    print("\nModel download complete!")
    print(f"Models saved in: {os.path.abspath(args.dir)}")
    print("You can now use these models locally in your Flask app.")
    
    return 0

if __name__ == "__main__":
    exit(main())