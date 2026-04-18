import os
from huggingface_hub import HfApi, hf_hub_download

token = os.getenv("HF_TOKEN", "")
repo_id = "m25csa001/ResearchCopilot"
local_file = "huggingface/app.py"

api = HfApi()

try:
    print("Uploading file...")
    commit_info = api.upload_file(
        path_or_fileobj=local_file,
        path_in_repo="app.py",
        repo_id=repo_id,
        repo_type="space",
        token=token,
        commit_message="Fix equation block rendering and normalize session history LaTeX"
    )
    print(f"Upload successful. Commit hash: {commit_info.oid}")

    print("Downloading file for verification...")
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename="app.py",
        repo_type="space",
        token=token,
        force_download=True
    )

    with open(downloaded_path, 'r') as f:
        content = f.read()
    
    check1 = 'content = _clean_report(content)' in content
    check2 = '_SUBSCRIPT_SIMPLE_RE' in content
    
    print(f"Verification - 'content = _clean_report(content)': {check1}")
    print(f"Verification - '_SUBSCRIPT_SIMPLE_RE': {check2}")

except Exception as e:
    print(f"Error: {e}")

