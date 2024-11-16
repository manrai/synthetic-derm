import os
from huggingface_hub import HfApi

api = HfApi()
webdataset_path = '/n/data1/hms/dbmi/manrai/derm/synderm2024/complete_10k_webdataset_shards/'
repo_id = 'tbuckley/synthetic-derm-dev-2'
repo_type = 'dataset' 

for root, dirs, files in os.walk(webdataset_path):
    for file in files:
        file_path = os.path.join(root, file)

        # Define the path in the repo relative to the repo root
        relative_path = os.path.relpath(file_path, webdataset_path)
        repo_path = os.path.join('data', relative_path).replace('\\', '/')
        
        print(f'Uploading {file_path} to {repo_path}')
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type=repo_type
        )