import asyncio
import websockets
import json
from PIL import Image
import io

# Function to resize the image
def resize_image(image_path, max_size=(400, 400)):
    image = Image.open(image_path)
    image.thumbnail(max_size)

    # Convert resized image to bytes
    buffer = io.BytesIO()
    image.save(buffer, format=image.format)  # Save in the original format (PNG/JPEG)
    return buffer.getvalue(), image.size, image.format

# Function to send image via WebSocket and handle responses
async def send_image(image_path):
    protocol = "wss" if input("Are you using HTTPS? (yes/no): ").lower() == "yes" else "ws"
    uri = f"{protocol}://localhost:8000/ws"  # WebSocket server URL

    try:
        # Establish WebSocket connection
        async with websockets.connect(uri) as websocket:
            # Resize the image and convert to bytes
            image_bytes, resized_size, image_format = resize_image(image_path)

            # Send the resized image
            await websocket.send(image_bytes)
            print(f"Image sent to the server: {resized_size}, Format: {image_format}")

            # Wait for server response
            response = await websocket.recv()
            data = json.loads(response)

            # Save response to a JSON file
            if 'objects' in data:
                with open('response.json', 'w') as json_file:
                    json.dump(data, json_file, indent=4)
                print('Response saved to response.json')
            else:
                print("Unexpected response format:", data)

    except websockets.exceptions.ConnectionClosedError as e:
        print(f"WebSocket connection closed unexpectedly: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Main entry point for the client
if __name__ == "__main__":
    image_path = input("Enter the path of the image file: ")  # Get image path from user
    asyncio.get_event_loop().run_until_complete(send_image(image_path))
