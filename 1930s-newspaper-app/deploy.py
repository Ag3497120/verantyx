import os
from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN")
api = HfApi(token=token)

username = api.whoami()["name"]
repo_id = f"{username}/TalkiePress"

print(f"Creating Space repo: {repo_id}...")
try:
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="gradio")
    print("Space created successfully.")
except Exception as e:
    print(f"Space might already exist or error: {e}")

print("Uploading files...")
api.upload_folder(
    folder_path="./",
    repo_id=repo_id,
    repo_type="space",
    ignore_patterns=["deploy.py"]
)
print(f"Upload complete! View your space at: https://huggingface.co/spaces/{repo_id}")
