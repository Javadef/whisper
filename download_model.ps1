# Download LLM Model Script for Windows

Write-Host "Whisper AI - Model Downloader" -ForegroundColor Cyan
Write-Host "==============================" -ForegroundColor Cyan
Write-Host ""

# Check if models directory exists
if (-not (Test-Path "models")) {
    New-Item -ItemType Directory -Path "models"
}

Write-Host "Available Models:" -ForegroundColor Yellow
Write-Host "1. Llama 3.2 3B Instruct (2GB) - Recommended for RTX 4060"
Write-Host "2. Qwen 2.5 7B Instruct (4.4GB) - Better multilingual support"
Write-Host "3. Gemma 2 9B IT (5.4GB) - Highest quality"
Write-Host ""

$choice = Read-Host "Select model to download (1-3)"

switch ($choice) {
    "1" {
        $url = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
        $filename = "llama-3.2-3b-instruct.Q4_K_M.gguf"
        $size = "~2 GB"
    }
    "2" {
        $url = "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf"
        $filename = "qwen2.5-7b-instruct.Q4_K_M.gguf"
        $size = "~4.4 GB"
    }
    "3" {
        $url = "https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/resolve/main/gemma-2-9b-it-Q4_K_M.gguf"
        $filename = "gemma-2-9b-it.Q4_K_M.gguf"
        $size = "~5.4 GB"
    }
    default {
        Write-Host "Invalid choice. Exiting." -ForegroundColor Red
        exit
    }
}

$outputPath = "models\$filename"

Write-Host ""
Write-Host "Downloading $filename ($size)..." -ForegroundColor Green
Write-Host "This may take a while depending on your internet speed." -ForegroundColor Yellow
Write-Host ""

# Download with progress
try {
    $ProgressPreference = 'SilentlyContinue'
    Invoke-WebRequest -Uri $url -OutFile $outputPath
    $ProgressPreference = 'Continue'
    
    Write-Host ""
    Write-Host "Download complete!" -ForegroundColor Green
    Write-Host "Model saved to: $outputPath" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "1. Update app/config.py with:" -ForegroundColor White
    Write-Host "   LLM_MODEL_PATH = MODELS_DIR / `"$filename`"" -ForegroundColor Gray
    Write-Host "2. Run: python run.py" -ForegroundColor White
    Write-Host ""
} catch {
    Write-Host ""
    Write-Host "Download failed: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Manual download:" -ForegroundColor Yellow
    Write-Host "URL: $url" -ForegroundColor Gray
    Write-Host "Save to: $outputPath" -ForegroundColor Gray
}
