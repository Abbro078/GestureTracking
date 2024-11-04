from aiohttp import web
import asyncio
import json
import logging
import cv2
import numpy as np
from PIL import Image
import io

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class WebRTCServer:
    def __init__(self):
        self.clients = set()
        self.frame_count = 0

    async def handle_camera_frame(self, binary_data):
        try:
            # Convert binary data to image
            image_stream = io.BytesIO(binary_data)
            pil_image = Image.open(image_stream)

            # Convert PIL image to OpenCV format for analysis
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            # Update frame count and log detailed info
            self.frame_count += 1
            height, width = cv_image.shape[:2]
            logger.info(f"Frame {self.frame_count}: Size={width}x{height}, Data size={len(binary_data) / 1024:.2f}KB")

            return True
        except Exception as e:
            logger.error(f"Error processing camera frame: {str(e)}")
            return False

    async def websocket_handler(self, request):
        logger.info("New client attempting to connect")
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        self.clients.add(ws)
        logger.info("Client connected")

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        logger.info(f"Received message: {data['type']}")

                        if data['type'] == 'offer':
                            if 'message' in data:
                                logger.info(f"Received test offer: {data['message']}")
                            answer = {
                                'type': 'answer',
                                'sdp': "Mock SDP for testing"
                            }
                            await ws.send_json(answer)
                            logger.info("Sent answer back to client")

                        elif data['type'] == 'candidate':
                            await ws.send_json({
                                'type': 'candidate',
                                'candidate': data['candidate']
                            })
                            logger.info("Echoed ICE candidate back to client")

                    except json.JSONDecodeError:
                        logger.error("Failed to parse message as JSON")
                    except KeyError as e:
                        logger.error(f"Missing key in message: {e}")

                elif msg.type == web.WSMsgType.BINARY:
                    success = await self.handle_camera_frame(msg.data)
                    if success:
                        logger.info("Successfully processed camera frame")
                    else:
                        logger.error("Failed to process camera frame")

                elif msg.type == web.WSMsgType.ERROR:
                    logger.error(f"WebSocket connection closed with exception {ws.exception()}")

        finally:
            self.clients.remove(ws)
            logger.info("Client disconnected")

        return ws


async def main():
    server = WebRTCServer()
    app = web.Application()
    app.router.add_get('/', server.websocket_handler)

    logger.info("Starting server...")

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8080)
    await site.start()

    logger.info("Server started at ws://0.0.0.0:8080")

    while True:
        await asyncio.sleep(3600)


if __name__ == '__main__':
    asyncio.run(main())