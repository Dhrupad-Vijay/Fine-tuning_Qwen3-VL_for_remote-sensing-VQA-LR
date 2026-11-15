import zipfile
import requests
from tqdm import tqdm
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

PTH = Path(__file__).resolve().parent

# Only what transformers needs. Skips duplicate weight formats (.bin, .h5, .msgpack).
MODEL_FILES = ["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.md", "LICENSE*"]

def download_RS_data(data_pth):
    api = f"https://zenodo.org/api/records/6344334"
    record = requests.get(api, timeout=60).json()

    for f in tqdm(record['files']):
        name, dest = f['key'], data_pth / f['key']
        dlink = f['links']['self']
        # print(f"Downloading {name}")

        with requests.get(dlink, stream=True, timeout=150) as r:
            r.raise_for_status()

            with open(dest, "wb") as file:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
        # print(f"Finished {f['key']}")

    images_zip = data_pth / "Images_LR.zip"
    if images_zip.exists():
        with zipfile.ZipFile(images_zip, 'r') as zip_ref:
            zip_ref.extractall(data_pth)
        images_zip.unlink()
    print("Dataset download complete\nModel downloads started...")

def download_models(model_pth, models):
    for model in models:
        sha = HfApi().model_info(model).sha
        dest = model_pth / model.split("/")[1]
        print(f"[model] {model} @ {sha[:8]} -> {dest.relative_to(PTH)}", flush=True)
        snapshot_download(model, revision=sha, local_dir=dest, allow_patterns=MODEL_FILES)
        # return {"repo": model, "revision": sha,
        #         "path": str(dest.relative_to(PTH))}

def main():
    data_pth = PTH / "data/rsvqa_lr"
    data_pth.mkdir(parents=True, exist_ok=True)
    model_pth = PTH / "models"

    # Download RSVQA low resolution datset
    download_RS_data(data_pth)

    # Download Qwen VL models 2B, 4B, 8B params models
    models = ["Qwen/Qwen3-VL-2B-Instruct",
              "Qwen/Qwen3-VL-4B-Instruct",
              "Qwen/Qwen3-VL-8B-Instruct"]
    download_models(model_pth, models)
    
if __name__ == "__main__":
    main()