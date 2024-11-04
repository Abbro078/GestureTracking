from aiohttp import web
import asyncio
import json
import ssl
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class WebRTCServer:
    def __init__(self):
        self.clients = set()

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
                            # Check if the offer contains a message
                            if 'message' in data:
                                logger.info(f"Received test offer: {data['message']}")
                            # Create and send back a mock answer
                            answer = {
                                'type': 'answer',
                                'sdp': "Mock SDP for testing"  # Just a placeholder
                            }
                            await ws.send_json(answer)
                            logger.info("Sent answer back to client")


                        elif data['type'] == 'candidate':
                            # Echo back the ICE candidate for testing
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
                    # Handle and print binary data (e.g., camera frames)
                    logger.info("Received binary data (camera frame)")
                    # For simplicity, we'll just log the size of the data received
                    logger.info(f"Binary data size: {len(msg.data)} bytes")
                    logger.info(f"Binary data: {msg.data}")


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

    # For development, use HTTP
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8080)
    await site.start()

    logger.info("Server started at ws://0.0.0.0:8080")

    # Keep the server running
    while True:
        await asyncio.sleep(3600)  # Sleep for an hour

if __name__ == '__main__':
    asyncio.run(main())