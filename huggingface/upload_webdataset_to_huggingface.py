import os
import argparse
from huggingface_hub import HfApi

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Upload webdataset shards to HuggingFace')
    parser.add_argument('--webdataset_path', type=str, required=True,
                        help='Path to webdataset shards directory')
    parser.add_argument('--repo_id', type=str, required=True,
                        help='HuggingFace repository ID (e.g. username/repo-name)')
    parser.add_argument('--repo_type', type=str, default='dataset',
                        help='Repository type (default: dataset)')

    args = parser.parse_args()

    api = HfApi()

    for root, dirs, files in os.walk(args.webdataset_path):
        for file in files:
            file_path = os.path.join(root, file)

            # Define the path in the repo relative to the repo root
            relative_path = os.path.relpath(file_path, args.webdataset_path)
            repo_path = os.path.join('data', relative_path).replace('\\', '/')
            
            print(f'Uploading {file_path} to {repo_path}')
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=repo_path,
                repo_id=args.repo_id,
                repo_type=args.repo_type
            )

