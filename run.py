"""Run the Whisper API server."""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable for production
        workers=1,  # Single worker for GPU (model sharing)
        log_level="info",
    )
