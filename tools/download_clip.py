import os
import argparse
import sys

def download_huggingface(repo_id, output_dir):
    try:
        from huggingface_hub import snapshot_download
        print(f"Starting download from HuggingFace: {repo_id}")
        
        model_name = repo_id.split('/')[-1]
        target_dir = os.path.join(output_dir, model_name)
        
        print(f"Downloading directly to: {target_dir}")
        path = snapshot_download(repo_id=repo_id, local_dir=target_dir, local_dir_use_symlinks=False)
        print(f"Download complete! Location: {path}")
        return True
    except Exception as e:
        print(f"Error downloading from HuggingFace: {e}")
        return False

def download_modelscope(model_id, output_dir):
    try:
        from modelscope.hub.snapshot_download import snapshot_download
        print(f"Starting download from ModelScope: {model_id}")
        
        model_name = model_id.split('/')[-1]
        target_dir = os.path.join(output_dir, model_name)
        
        print(f"Downloading directly to: {target_dir}")
        # Try using local_dir if supported (newer modelscope versions)
        try:
            path = snapshot_download(model_id=model_id, local_dir=target_dir)
        except TypeError:
            # Fallback for older versions that don't support local_dir
            print("ModelScope version does not support local_dir, falling back to cache download...")
            path = snapshot_download(model_id=model_id)
            import shutil
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            shutil.copytree(path, target_dir)
            print("Moved model to target directory.")
            
        print(f"Download complete! Location: {target_dir}")
        return True
    except Exception as e:
        print(f"Error downloading from ModelScope: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download CLIP model")
    parser.add_argument("--source", type=str, choices=["huggingface", "modelscope"], default="huggingface", help="Download source")
    parser.add_argument("--model", type=str, default="openai/clip-vit-base-patch32", help="Model ID")
    parser.add_argument("--output-dir", type=str, default="tools/filter_style", help="Output directory")

    args = parser.parse_args()
    
    # Ensure output dir exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    success = False
    if args.source == "modelscope":
        success = download_modelscope(args.model, args.output_dir)
    else:
        success = download_huggingface(args.model, args.output_dir)

    if not success:
        sys.exit(1)
