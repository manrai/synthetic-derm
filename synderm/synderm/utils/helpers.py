from pathlib import Path
import os

def save_images(images, directory):
    pass

def load_images(directory):
    pass

def count_png_files(directory_path):
    """
    Count the total number of PNG files in a directory and its subdirectories
    
    Args:
        directory_path (str): Path to the directory to search
        
    Returns:
        int: Total number of PNG files found
    """
    total_png = 0
    for root, dirs, files in os.walk(directory_path, followlinks=True):
        png_files = [f for f in files if f.endswith('.png')]
        total_png += len(png_files)
    return total_png

def save(image, path):
    path = Path(path) if isinstance(path, str) else path
    path.parent.mkdir(exist_ok=True, parents=True)
    image.save(path)

