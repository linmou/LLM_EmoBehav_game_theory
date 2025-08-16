#!/usr/bin/env python3
"""
Long Input Handling Demo

Demonstrates the server's ability to handle various input lengths
and provides examples of performance testing.
"""

import time
import requests
import json
from typing import Dict, Any


class LongInputDemo:
    """Demo class for long input handling"""
    
    def __init__(self, base_url: str = "http://localhost:8000/v1"):
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer dummy"
        }

    def generate_sample_text(self, length: int) -> str:
        """Generate sample text of specified length"""
        base_text = (
            "Artificial intelligence and machine learning are rapidly transforming "
            "how we interact with technology and solve complex problems. "
            "These technologies enable computers to learn from data, recognize patterns, "
            "and make decisions with minimal human intervention. "
        )
        
        repetitions = (length // len(base_text)) + 1
        return (base_text * repetitions)[:length]

    def test_basic_request(self, input_length: int) -> Dict[str, Any]:
        """Test a basic request with specified input length"""
        text = self.generate_sample_text(input_length)
        
        payload = {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": f"Please summarize this text: {text}"}
            ],
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        print(f"📤 Sending request with {input_length} character input...")
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                response_content = data["choices"][0]["message"]["content"]
                
                result = {
                    "success": True,
                    "input_length": input_length,
                    "response_time": response_time,
                    "output_length": len(response_content),
                    "response_preview": response_content[:100] + "..." if len(response_content) > 100 else response_content,
                    "tokens_used": data.get("usage", {}).get("total_tokens", "N/A")
                }
                
                print(f"✅ Success! Response time: {response_time:.2f}s")
                print(f"   Output length: {len(response_content)} chars")
                print(f"   Response preview: {result['response_preview']}")
                print()
                
                return result
            
            else:
                print(f"❌ Request failed with status {response.status_code}")
                print(f"   Error: {response.text}")
                print()
                
                return {
                    "success": False,
                    "input_length": input_length,
                    "response_time": response_time,
                    "error": response.text,
                    "status_code": response.status_code
                }
        
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            
            print(f"❌ Request failed with exception: {e}")
            print()
            
            return {
                "success": False,
                "input_length": input_length,
                "response_time": response_time,
                "error": str(e)
            }

    def test_streaming_request(self, input_length: int) -> Dict[str, Any]:
        """Test a streaming request with specified input length"""
        text = self.generate_sample_text(input_length)
        
        payload = {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": f"Please explain this topic in detail: {text}"}
            ],
            "max_tokens": 150,
            "temperature": 0.7,
            "stream": True
        }
        
        print(f"🌊 Testing streaming with {input_length} character input...")
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                stream=True,
                timeout=30
            )
            
            if response.status_code == 200:
                chunks = []
                first_chunk_time = None
                
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith("data: "):
                            data_str = line_str[6:]
                            if data_str.strip() != "[DONE]":
                                try:
                                    chunk_data = json.loads(data_str)
                                    if first_chunk_time is None:
                                        first_chunk_time = time.time()
                                    chunks.append(chunk_data)
                                except json.JSONDecodeError:
                                    continue
                
                end_time = time.time()
                total_time = end_time - start_time
                time_to_first_chunk = first_chunk_time - start_time if first_chunk_time else None
                
                # Reconstruct full response
                full_response = ""
                for chunk in chunks:
                    if chunk.get("choices", [{}])[0].get("delta", {}).get("content"):
                        full_response += chunk["choices"][0]["delta"]["content"]
                
                print(f"✅ Streaming success!")
                print(f"   Total chunks received: {len(chunks)}")
                print(f"   Time to first chunk: {time_to_first_chunk:.2f}s" if time_to_first_chunk else "   Time to first chunk: N/A")
                print(f"   Total streaming time: {total_time:.2f}s")
                print(f"   Full response length: {len(full_response)} chars")
                print()
                
                return {
                    "success": True,
                    "input_length": input_length,
                    "total_time": total_time,
                    "time_to_first_chunk": time_to_first_chunk,
                    "chunks_count": len(chunks),
                    "response_length": len(full_response)
                }
            
            else:
                print(f"❌ Streaming failed with status {response.status_code}")
                print(f"   Error: {response.text}")
                print()
                
                return {
                    "success": False,
                    "input_length": input_length,
                    "error": response.text,
                    "status_code": response.status_code
                }
        
        except Exception as e:
            print(f"❌ Streaming failed with exception: {e}")
            print()
            
            return {
                "success": False,
                "input_length": input_length,
                "error": str(e)
            }

    def run_demo(self):
        """Run the complete demo"""
        print("🚀 Long Input Handling Demo")
        print("=" * 50)
        print()
        
        # Check server health
        try:
            health_url = self.base_url.replace("/v1", "/health")
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                print("✅ Server health check passed")
            else:
                print(f"⚠️  Server health check returned status {response.status_code}")
        except Exception as e:
            print(f"❌ Server health check failed: {e}")
            print("Please ensure the server is running on the expected port.")
            return
        
        print()
        
        # Test different input lengths
        test_lengths = [100, 500, 1000, 2000, 5000]
        
        print("📊 Testing Progressive Input Lengths")
        print("-" * 40)
        
        for length in test_lengths:
            result = self.test_basic_request(length)
            time.sleep(1)  # Brief pause between requests
        
        print("🌊 Testing Streaming with Long Input")
        print("-" * 40)
        
        # Test streaming with a moderately long input
        streaming_result = self.test_streaming_request(2000)
        
        print("✨ Demo completed!")
        print()
        print("💡 Tips for Testing:")
        print("   - Use shorter inputs for faster testing")
        print("   - Monitor server logs for detailed information")
        print("   - Test concurrent requests for load testing")
        print("   - Adjust max_tokens based on your needs")


def main():
    """Run the demo"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Long Input Handling Demo")
    parser.add_argument("--server-url", 
                       default="http://localhost:8000/v1",
                       help="OpenAI server URL")
    
    args = parser.parse_args()
    
    demo = LongInputDemo(args.server_url)
    demo.run_demo()


if __name__ == "__main__":
    main()