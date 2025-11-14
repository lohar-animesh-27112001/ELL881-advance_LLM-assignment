import os
import requests
import gzip
import shutil

def download_fasttext_model(save_dir="./", filename="cc.en.300.bin"):
    """
    Download and extract the FastText English model (cc.en.300.bin).
    """
    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz"
    compressed_file = os.path.join(save_dir, filename + ".gz")
    extracted_file = os.path.join(save_dir, filename)
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Download the .gz file
    print(f"Downloading {url} ...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024  # 1MB chunks
    
    with open(compressed_file, "wb") as f:
        for data in response.iter_content(block_size):
            f.write(data)
    print(f"Download completed: {compressed_file}")
    
    # Extract the .gz file
    print(f"Extracting {compressed_file} ...")
    with gzip.open(compressed_file, 'rb') as f_in:
        with open(extracted_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Extraction completed: {extracted_file}")
    
    # Optionally, remove the .gz file to save space
    os.remove(compressed_file)
    print(f"Removed compressed file: {compressed_file}")
    
    return extracted_file

# Example usage
if __name__ == "__main__":
    # model_path = download_fasttext_model(save_dir="./")
    # print(f"FastText model saved at: {model_path}")
    import fasttext
    model = fasttext.load_model("./fasttext/cc.en.300.bin")
    print(model.get_word_vector("hello I am Animesh Lohar"))

