#!/usr/bin/env python3
"""WebSocket client to test the server with proper Unicode handling"""

import asyncio
import json
import websockets

async def test_websocket():
    uri = "ws://localhost:8000"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to WebSocket server")
            
            # Send a test message
            message = {
                "question": "hi",
                "history": [["Q1", "A1"]],
                "return_sources": True
            }
            
            await websocket.send(json.dumps(message))
            print(f"Sent: {message}")
            
            # Receive and display responses
            full_answer = ""
            while True:
                try:
                    response = await websocket.recv()
                    data = json.loads(response)
                    
                    if data["type"] == "token":
                        content = data["content"]
                        print(content, end="", flush=True)
                        full_answer += content
                    elif data["type"] == "done":
                        print(f"\n\nFull answer: {data['answer']}")
                        if "sources" in data:
                            print(f"Sources: {len(data['sources'])} chunks")
                        break
                    elif data["type"] == "error":
                        print(f"\nError: {data['message']}")
                        break
                        
                except websockets.exceptions.ConnectionClosed:
                    print("\nConnection closed")
                    break
                except json.JSONDecodeError as e:
                    print(f"\nFailed to parse JSON: {e}")
                    print(f"Raw response: {response}")
                    break
                    
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())