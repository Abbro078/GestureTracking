# import argparse
# import cv2 as cv
# import mp
# import zmq
# import numpy as np
# import logging
# import tensorflow as tf
# from keras import models
# from mediapipe.python.solutions import hands
# import mediapipe as mp
# from tensorflow.python.ops.logging_ops import Print
#
#
# model = models.load_model('model/keypoint_classifier/keypoint_classifier.keras')
# context = zmq.Context()
# socket = context.socket(zmq.REP)
# socket.bind('tcp://*:5555')
#
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)
#
# def get_args():
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument("--device", type=int, default=0)
#     parser.add_argument("--width", help='cap width', type=int, default=960)
#     parser.add_argument("--height", help='cap height', type=int, default=540)
#
#     parser.add_argument('--use_static_image_mode', action='store_true')
#     parser.add_argument("--min_detection_confidence",
#                         help='min_detection_confidence',
#                         type=float,
#                         default=0.7)
#     parser.add_argument("--min_tracking_confidence",
#                         help='min_tracking_confidence',
#                         type=int,
#                         default=0.5)
#
#     args = parser.parse_args()
#
#     return args
#
#
# args = get_args()
#
# use_static_image_mode = args.use_static_image_mode
# min_detection_confidence = args.min_detection_confidence
# min_tracking_confidence = args.min_tracking_confidence
#
# while True:
#     image_bytes = socket.recv()
#     print(image_bytes)
#     socket.send_string("othman wasnt hungry enough")
#     logger.info("sent")
#
#     nparr = np.frombuffer(image_bytes, np.uint8)
#     print(nparr.shape)
#     mp_hands = mp.solutions.hands
#     hands = mp_hands.Hands(
#         static_image_mode=use_static_image_mode,
#         max_num_hands=2,
#         min_detection_confidence=min_detection_confidence,
#         min_tracking_confidence=min_tracking_confidence,
#     )
#
#     image = cv.imdecode(nparr, cv.IMREAD_COLOR)
#     if image is None:
#         print("Failed to decode image")
#         continue
#
#     image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
#
#     # image_rgb.flags.writeable = False
#     # results = hands.process(image_rgb)
#     # image_rgb.flags.writeable = True
#     #
#     # if results.multi_hand_landmarks:
#     #     for hand_landmarks in results.multi_hand_landmarks:
#     #         mp.solutions.drawing_utils.draw_landmarks(
#     #             image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#
#     cv.imshow("Received Image", image_rgb)
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break
#
#     '''array = np.frombuffer(bytes, dtype=np.uint8)
#     pred = model.predict(array)
#     bytes_to_send = pred.tobytes()
#     socket.send(bytes_to_send)'''


import argparse
import cv2 as cv
import zmq
import numpy as np
import logging
import mediapipe as mp
from mediapipe.python.solutions import hands

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", help='cap width', type=int, default=640)
    parser.add_argument("--height", help='cap height', type=int, default=480)
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=int, default=0.5)
    return parser.parse_args()


def initialize_zmq():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind('tcp://*:5555')
    return socket


def process_image(image, mp_hands, hands_detector):
    if image is None:
        logger.error("No image to process")
        return None

    # Convert the image
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Process the image
    image_rgb.flags.writeable = False
    results = hands_detector.process(image_rgb)
    image_rgb.flags.writeable = True

    # Draw the hand annotations on the image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return cv.cvtColor(image_rgb, cv.COLOR_RGB2BGR)


def main():
    args = get_args()
    socket = initialize_zmq()

    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        static_image_mode=args.use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    logger.info("Server started, waiting for connections...")

    try:
        while True:
            # Receive image data
            image_bytes = socket.recv()
            logger.debug(f"Received image data: {len(image_bytes)} bytes")

            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)

            # Decode image
            image = cv.imdecode(nparr, cv.IMREAD_COLOR)
            if image is None:
                logger.error("Failed to decode image")
                socket.send_string("error: failed to decode image")
                continue

            # Process image
            processed_image = process_image(image, mp_hands, hands_detector)
            if processed_image is not None:
                # Show the image
                cv.imshow("Received Image", processed_image)

                # Send response
                socket.send_string("Image processed successfully")
            else:
                socket.send_string("error: failed to process image")

            # Check for quit
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    finally:
        cv.destroyAllWindows()
        socket.close()


if __name__ == "__main__":
    main()