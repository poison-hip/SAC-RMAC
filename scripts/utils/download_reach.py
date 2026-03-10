from huggingface_hub import hf_hub_download
import os

repo_id = "chloezandberg/sac-PandaReachDense-v3"
filename = "sac-PandaReachDense-v3.zip"
target_dir = "logs"

print(f"Downloading {filename} from {repo_id}...")
try:
    path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=target_dir)
    print(f"Success! Downloaded to {path}")
except Exception as e:
    print(f"Error downloading {filename}: {e}")
    # Try alternate filename 'best_model.zip' which is common
    print("Trying best_model.zip...")
    try:
        path = hf_hub_download(repo_id=repo_id, filename="best_model.zip", local_dir=target_dir)
        # Rename to expected name for consistency
        new_path = os.path.join(target_dir, filename)
        os.rename(path, new_path)
        print(f"Success! Downloaded best_model.zip and renamed to {new_path}")
    except Exception as e2:
        print(f"Error downloading best_model.zip: {e2}")
