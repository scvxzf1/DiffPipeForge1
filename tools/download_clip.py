import os
import sys
import argparse

def download_huggingface(repo_id):
    try:
        from huggingface_hub import snapshot_download
        print(f"Starting download from HuggingFace: {repo_id}")
        path = snapshot_download(repo_id=repo_id)
        print(f"Download complete! Model saved at: {path}")
        return True
    except Exception as e:
        print(f"Error downloading from HuggingFace: {e}")
        return False

def download_modelscope(model_id):
    try:
        from modelscope.hub.snapshot_download import snapshot_download
        print(f"Starting download from ModelScope: {model_id}")
        path = snapshot_download(model_id=model_id)
        print(f"Download complete! Model saved at: {path}")
        return True
    except Exception as e:
        print(f"Error downloading from ModelScope: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download CLIP model from ModelScope or HuggingFace")
    parser.add_argument("--source", type=str, choices=["huggingface", "modelscope"], default="huggingface", help="Download source")
    parser.add_argument("--model", type=str, default="openai/clip-vit-base-patch32", help="Model ID to download")
    
    args = parser.parse_all() if hasattr(parser, 'parse_all') else parser.parse_args()
    
    success = False
    if args.source == "modelscope":
        success = download_modelscope(args.model)
    else:
        success = download_huggingface(args.model)
    
    if not success:
        sys.exit(1)
