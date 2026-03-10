from huggingface_hub import hf_hub_download, list_repo_files
import os

repos = [
    "Hevagog/sac-PandaSlide-v3",
    "chloezandberg/sac-PandaReachDense-v3"
]

target_dir = "logs"
os.makedirs(target_dir, exist_ok=True)

for repo_id in repos:
    print(f"Checking repo: {repo_id}")
    try:
        files = list_repo_files(repo_id)
        print(f"Files in {repo_id}: {files}")
        
        # Look for zip file
        zip_file = next((f for f in files if f.endswith(".zip")), None)
        if zip_file:
            print(f"Downloading {zip_file} from {repo_id}...")
            path = hf_hub_download(repo_id=repo_id, filename=zip_file, local_dir=target_dir)
            print(f"Downloaded to {path}")
            
            # Verify zip
            import zipfile
            try:
                with zipfile.ZipFile(path, 'r') as z:
                    if 'data' in z.namelist() or 'best_model.zip' in z.namelist() or any(f.endswith('.zip') for f in z.namelist()):
                        print(f"Verified {path} is a valid zip.")
                        exit(0) # Success!
                    else:
                        print(f"Warning: {path} contents look suspicious: {z.namelist()[:5]}")
            except Exception as e:
                print(f"Invalid zip: {e}")
        else:
            print(f"No zip file found in {repo_id}")

    except Exception as e:
        print(f"Error accessing {repo_id}: {e}")

print("Failed to download any valid model.")
exit(1)
