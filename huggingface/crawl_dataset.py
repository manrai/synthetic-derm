import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import random
import csv
import os

root_dir = "/n/data1/hms/dbmi/manrai/derm/synderm2024/complete_new"

entries = [entry for entry in os.scandir(root_dir) if entry.is_dir()]

counts = {}
for entry in entries:
    subfolder_count = 0

    with os.scandir(entry.path) as sub_it:
        file_bar = tqdm(desc=f"Processing files in {entry.name}")
        for f in sub_it:
            if f.is_file() and f.name.endswith('.png'):
                subfolder_count += 1
                file_bar.update(1)

    file_bar.close()  # Close the file progress bar
    count = subfolder_count
    print(f"Number of files found at {entry.name}: {count}")
    counts[entry.name] = count

csv_path = os.path.join('huggingface', 'folder_counts.csv')
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Folder', 'Image Count'])
    for folder, count in counts.items():
        writer.writerow([folder, count])
