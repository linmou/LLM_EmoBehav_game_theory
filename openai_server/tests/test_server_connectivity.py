#!/usr/bin/env python3
"""
Quick test script to check if OpenAI servers are running and accessible.
"""

import sys
from urllib.parse import urljoin

import requests


def test_server(port, emotion="neutral"):
    """Test if server is running on given port."""
    base_url = f"http://localhost:{port}/v1"

    print(f"\n{'='*60}")
    print(f"Testing {emotion.upper()} server on port {port}")
    print(f"{'='*60}")

    try:
        # Test models endpoint
        models_url = base_url + "/models" if not base_url.endswith("/") else base_url + "models"
        response = requests.get(models_url, timeout=5)
        if response.status_code == 200:
            print(f"✅ Server is running at {base_url}")
            data = response.json()
            if "data" in data and data["data"]:
                print(f"   Model: {data['data'][0].get('id', 'Unknown')}")
            return True
        else:
            print(f"❌ Server returned status {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to server at {base_url}")
        print(f"   Please start the server with:")
        if port == 8000:
            print(f"   python -m openai_server --model <model_path> --port 8000")
        else:
            print(
                f"   python -m openai_server --model <model_path> --emotion {emotion} --port {port}"
            )
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Test both neutral and anger servers."""
    print("OpenAI Server Connectivity Test")
    print("This script checks if the required servers are running.")

    # Test neutral server (port 8000)
    neutral_ok = test_server(8000, "neutral")

    # Test anger server (port 8001)
    anger_ok = test_server(8001, "anger")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Neutral Server (8000): {'✅ ONLINE' if neutral_ok else '❌ OFFLINE'}")
    print(f"Anger Server (8001):   {'✅ ONLINE' if anger_ok else '❌ OFFLINE'}")

    if neutral_ok and anger_ok:
        print("\n🎉 Both servers are running! Ready for experiments.")
        sys.exit(0)
    else:
        print("\n❌ One or more servers are not running.")
        print("Please start the missing servers before running experiments.")
        sys.exit(1)


if __name__ == "__main__":
    main()
