import argparse
import os
from pathlib import Path
from huggingface_hub import snapshot_download

def download_model(repo_id, local_dir):
    print(f"Downloading {repo_id} to {local_dir}...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.git*"]
    )
    print("Download complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HF model weights")
    parser.add_argument("--repo-id", type=str, default="codellama/CodeLlama-7b-hf", help="HF Repo ID")
    parser.add_argument("--folder-name", type=str, default="codellama-7b", help="Folder name in checkpoints/")
    args = parser.parse_args()

    # Determine checkpoints dir relative to this script or current dir
    # Assuming run from root or scripts/
    base_dir = Path("checkpoints")
    if not base_dir.exists():
        # Try going up one level if we are in scripts/
        if Path("../checkpoints").exists():
            base_dir = Path("../checkpoints")
        else:
            base_dir.mkdir(exist_ok=True)
            
    target_dir = base_dir / args.folder_name
    download_model(args.repo_id, target_dir)
