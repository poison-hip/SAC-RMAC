import requests
import os

candidates = [
    ("FetchSlide-v1", "https://huggingface.co/sb3/sac-FetchSlide-v1/resolve/main/sac-FetchSlide-v1.zip"),
    ("FetchPush-v1", "https://huggingface.co/sb3/sac-FetchPush-v1/resolve/main/sac-FetchPush-v1.zip"),
    ("PandaReachDense-v3", "https://huggingface.co/ppb/sac-PandaReachDense-v3/resolve/main/sac-PandaReachDense-v3.zip"),
    ("PandaSlide-v3", "https://huggingface.co/ppb/sac-PandaSlide-v3/resolve/main/sac-PandaSlide-v3.zip"),
    ("PandaPickAndPlace-v3", "https://huggingface.co/SamuelBernardDev/sac-PandaPickAndPlace-v3/resolve/main/sac-PandaPickAndPlace-v3.zip"),

]

target_dir = "logs"
os.makedirs(target_dir, exist_ok=True)

for name, url in candidates:
    print(f"Trying to download {name} from {url}...")
    try:
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            filename = os.path.join(target_dir, f"sac-{name}.zip")
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Success! Downloaded to {filename}")
            
            # Verify zip
            import zipfile
            try:
                with zipfile.ZipFile(filename, 'r') as z:
                    if 'data' in z.namelist() or 'best_model.zip' in z.namelist() or any(f.endswith('.zip') for f in z.namelist()):
                        print(f"Verified {filename} is a valid zip.")
                        exit(0)
                    else:
                        print(f"Warning: {filename} does not contain expected model files.")
            except zipfile.BadZipFile:
                print(f"Error: {filename} is not a valid zip file.")
                os.remove(filename)
        else:
            print(f"Failed with status code: {r.status_code}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

print("All attempts failed.")
exit(1)
