# Sets up the isolated voice-clone environment (.venv-voice) used by
# censor_mode=voice_clone. Needs an NVIDIA GPU with >= 4 GB VRAM for
# real-time use (CPU works but is far too slow for live calls).
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup_voice_clone.ps1
#
# Downloads ~3 GB of packages and ~4 GB of model weights on first run.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$base = & "$root\venv\Scripts\python.exe" -c "import sys; print(sys.base_prefix)"
if (-not (Test-Path ".venv-voice")) {
    Write-Host "Creating .venv-voice from $base"
    & "$base\python.exe" -m venv .venv-voice
}
$py = "$root\.venv-voice\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
& $py -m pip install -r "$root\voice_service\requirements.txt"

Write-Host "`nDownloading Chatterbox Turbo weights and checking the GPU..."
& $py -c "import torch; print('CUDA available:', torch.cuda.is_available()); from huggingface_hub import snapshot_download; snapshot_download('ResembleAI/chatterbox-turbo', allow_patterns=['*.safetensors','*.json','*.txt','*.pt','*.model'])"
Write-Host "`nDone. Start SpeechShield with:  python main.py --mode interceptor --censor-mode voice_clone"
