import os
import requests
from tqdm import tqdm

def download_file_robust(url, filename):
    """Downloads a file in chunks to prevent corruption/timeouts."""
    if os.path.exists(filename):
        if "pth.tar" in filename and os.path.getsize(filename) < 90 * 1024 * 1024:
            print(f"File {filename} looks corrupt (too small). Re-downloading...")
            os.remove(filename)
        else:
            return

    print(f"Downloading {filename}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))

        with open(filename, "wb") as file, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
        print("Download complete.")
    except Exception as e:
        print(f"Download failed: {e}")
        if os.path.exists(filename):
            os.remove(filename)

def download_scene_hierarchy_file():
    """Downloads the scene hierarchy CSV file from Google Docs."""
    url = 'https://docs.google.com/spreadsheet/ccc?key=1H7ADoEIGgbF_eXh9kcJjCs5j_r3VJwke4nebhkdzksg&output=csv'
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open("scene_hierarchy_places365.csv", "w", encoding='utf-8') as file:
            file.write(response.content.decode('utf-8') + "\n")
    except Exception as e:
        print(f"Error downloading hierarchy CSV: {e}")

def download_pretrained_on_places(model_name="resnet50"):
    """Downloads a pre-trained Places365 model."""
    url = f'http://places2.csail.mit.edu/models_places365/{model_name}_places365.pth.tar'
    download_file_robust(url, f"{model_name}_places365.pth.tar")