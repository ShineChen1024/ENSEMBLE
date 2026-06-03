#!/bin/bash
export PATH="/home/work/MMSearch/cwf/miniconda/envs/search-train/bin:$PATH"

echo "[$(date)] Starting upload..."

python3 -u << 'PYEOF'
import httpcore
import ssl
print("Patching SSL...", flush=True)

_orig_connect = httpcore._sync.connection.HTTPConnection._connect
def _patched_connect(self, request):
    if self._origin.scheme == b"https":
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        self._ssl_context = ctx
    return _orig_connect(self, request)
httpcore._sync.connection.HTTPConnection._connect = _patched_connect

print("Importing huggingface_hub...", flush=True)
from huggingface_hub import login, upload_folder

print("Logging in...", flush=True)
login(token="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")

print("Starting upload...", flush=True)
upload_folder(
    folder_path="/home/work/MMSearch/cwf/ENSEMBLE/qwen_image_edit_lora_qkv",
    path_in_repo="qwen_image_edit_lora_qkv",
    repo_id="ShineChen1024/ENSEMBLE",
    repo_type="model",
)
print("Upload done!", flush=True)
PYEOF

echo "[$(date)] Script finished."
