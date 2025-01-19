import os
import requests
from tqdm import tqdm

def download_file(url: str, save_path: str, chunk_size: int = 8192) -> str:
    """
    Download a file from a URL and save it to the specified path.
    Shows a progress bar during download.

    Args:
        url: URL of the file to download
        save_path: Local path where the file should be saved
        chunk_size: Size of chunks to download at a time (bytes)

    Returns:
        str: Path to the downloaded file

    Raises:
        requests.exceptions.RequestException: If download fails
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    # Send GET request with stream=True to download in chunks
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Get total file size if available
    total_size = int(response.headers.get('content-length', 0))
    
    # Open file and write chunks with progress bar
    with open(save_path, 'wb') as f, \
         tqdm(desc=os.path.basename(save_path),
              total=total_size,
              unit='iB',
              unit_scale=True,
              unit_divisor=1024) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            size = f.write(chunk)
            pbar.update(size)

    # Check if downloaded file is a zip file
    if not save_path.endswith('.zip'):
        raise ValueError("Downloaded file must be a zip file")
        
    # Get target directory from save path
    target_dir = os.path.dirname(save_path)
    
    # Extract zip file
    import zipfile
    print(f"Extracting {save_path} to {target_dir}...")
    with zipfile.ZipFile(save_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

    # Remove zip file
    os.remove(save_path)
            

if __name__ == "__main__":
    url = "https://github.com/uni-medical/MedSegModelZoo/releases/download/test_0.0.1/Task001_BrainTumour.zip"
    save_path = "checkpoints/Task001_BrainTumour.zip"
    download_file(url, save_path)
