"""Test script for the Whisper API."""
import requests
import sys
from pathlib import Path

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"GPU: {data.get('gpu_name', 'Not Available')}")
    print(f"Whisper: {'✓' if data.get('models_loaded') else '✗'} ({data.get('whisper_model', 'N/A')})")
    print(f"LLM: {'✓' if data.get('llm_loaded') else '✗'}")
    return response.status_code == 200


def test_languages():
    """Test languages endpoint."""
    print("\nTesting languages endpoint...")
    response = requests.get(f"{BASE_URL}/languages")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Transcription: {data.get('transcription_languages')}")
    print(f"Translation: {', '.join(data.get('translation_languages', {}).keys())}")
    return response.status_code == 200


def test_chat():
    """Test chat endpoint."""
    print("\nTesting chat endpoint...")
    
    response = requests.post(
        f"{BASE_URL}/chat",
        json={
            "message": "What languages can you translate?",
            "session_id": "test-session"
        }
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Response: {data['response'][:200]}...")
        return True
    else:
        print(f"Error: {response.json()}")
        return False


def test_llm_translate():
    """Test LLM translation."""
    print("\nTesting LLM translation...")
    
    response = requests.post(
        f"{BASE_URL}/translate/llm",
        json={
            "text": "Salom, bugun ob-havo juda yaxshi.",
            "source_language": "uzbek",
            "target_language": "english"
        }
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Original: {data['source_text']}")
        print(f"Translated: {data['translated_text']}")
        return True
    else:
        print(f"Error: {response.json()}")
        return False


def test_transcribe(file_path: str, translate_to: str = None):
    """Test transcription endpoint."""
    print(f"\nTesting transcription with: {file_path}")
    
    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        return False
    
    with open(file_path, "rb") as f:
        files = {"file": (Path(file_path).name, f)}
        data = {"task": "transcribe", "word_timestamps": "false"}
        if translate_to:
            data["translate_to"] = translate_to
        
        response = requests.post(f"{BASE_URL}/transcribe", files=files, data=data)
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Detected language: {result['detected_language']}")
        print(f"Duration: {result['duration']:.2f}s")
        print(f"Processing time: {result['processing_time']:.2f}s")
        print(f"Transcription ({len(result['transcription'])} chars):")
        print(f"  {result['transcription'][:300]}...")
        
        if result.get('translations'):
            print(f"\nTranslations:")
            for lang, text in result['translations'].items():
                print(f"  {lang}: {text[:200]}...")
        
        return True
    else:
        print(f"Error: {response.json()}")
        return False


def test_async_transcribe(file_path: str):
    """Test async transcription endpoint."""
    print(f"\nTesting async transcription with: {file_path}")
    
    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        return False
    
    # Submit task
    with open(file_path, "rb") as f:
        files = {"file": (Path(file_path).name, f)}
        data = {"task": "transcribe"}
        response = requests.post(f"{BASE_URL}/transcribe/async", files=files, data=data)
    
    if response.status_code != 200:
        print(f"Failed to submit task: {response.json()}")
        return False
    
    task_id = response.json()["task_id"]
    print(f"Task ID: {task_id}")
    
    # Poll for result
    import time
    max_wait = 300  # 5 minutes
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        response = requests.get(f"{BASE_URL}/task/{task_id}")
        status = response.json()
        
        status_msg = status['status']
        progress = status.get('progress', 0)
        print(f"Status: {status_msg}, Progress: {progress:.1%}")
        
        if status["status"] == "completed":
            result = status['result']
            print(f"\nTranscription: {result['transcription'][:300]}...")
            return True
        elif status["status"] == "failed":
            print(f"Error: {status['error']}")
            return False
        
        time.sleep(2)
    
    print("Timeout waiting for result")
    return False


def test_stream_info():
    """Test stream info endpoint."""
    print("\nTesting stream info...")
    
    # Test with a known working stream
    test_url = "https://www.youtube.com/watch?v=jfKfPfyJRdk"  # Live stream example
    
    response = requests.get(f"{BASE_URL}/stream/info", params={"url": test_url})
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"URL: {data['url']}")
        print(f"Supported: {data['supported']}")
        print(f"Qualities: {', '.join(data.get('qualities', []))}")
        return True
    else:
        print(f"Error: {response.json()}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Whisper AI - Comprehensive Test Suite")
    print("=" * 60)
    
    results = []
    
    # Basic tests
    results.append(("Health Check", test_health()))
    results.append(("Languages", test_languages()))
    
    # LLM tests (may fail if model not loaded)
    try:
        results.append(("Chat", test_chat()))
        results.append(("LLM Translation", test_llm_translate()))
    except Exception as e:
        print(f"\nLLM tests skipped (model not loaded): {e}")
    
    # Stream info test
    try:
        results.append(("Stream Info", test_stream_info()))
    except Exception as e:
        print(f"\nStream test skipped: {e}")
    
    # File transcription test
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        translate_to = sys.argv[2] if len(sys.argv) > 2 else "english"
        
        results.append(("Transcription", test_transcribe(file_path, translate_to)))
        
        # Test async if sync works
        if results[-1][1]:
            results.append(("Async Transcription", test_async_transcribe(file_path)))
    else:
        print("\n" + "=" * 60)
        print("To test transcription, run:")
        print("  python test_api.py <video_or_audio_file> [translate_to]")
        print("  Example: python test_api.py video.mp4 english,russian")
        print("=" * 60)
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} - {test_name}")
    
    print("=" * 60)
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 60)

