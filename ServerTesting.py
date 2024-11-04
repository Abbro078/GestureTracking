import asyncio
import itertools
import json
import cv2
import numpy as np
import mediapipe as mp
from collections import deque, Counter
import copy
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole
from websockets.server import serve
from model import KeyPointClassifier
from model import PointHistoryClassifier
import csv
import socket



class HandGestureProcessor:
    def __init__(self):
        # MediaPipe initialization
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        # Classifier initialization
        self.keypoint_classifier = KeyPointClassifier()
        self.point_history_classifier = PointHistoryClassifier()

        # Load labels
        with open('model/keypoint_classifier/keypoint_classifier_label.csv',
                  encoding='utf-8-sig') as f:
            self.keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

        with open('model/point_history_classifier/point_history_classifier_label.csv',
                  encoding='utf-8-sig') as f:
            self.point_history_classifier_labels = [row[0] for row in csv.reader(f)]

        # Initialize history
        self.history_length = 16
        self.point_history = deque(maxlen=self.history_length)
        self.finger_gesture_history = deque(maxlen=self.history_length)

    async def process_frame(self, frame):
        # Convert frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is None:
            self.point_history.append([0, 0])
            return None, frame

        processed_results = []
        debug_image = copy.deepcopy(frame)

        for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                              results.multi_handedness):
            # Get landmark data
            landmark_list = self.calc_landmark_list(frame, hand_landmarks)

            # Process landmarks
            pre_processed_landmark_list = self.pre_process_landmark(landmark_list)
            pre_processed_point_history_list = self.pre_process_point_history(
                frame, self.point_history)

            # Classify hand gesture
            hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)

            if hand_sign_id == 2:  # Point gesture
                self.point_history.append(landmark_list[8])
            else:
                self.point_history.append([0, 0])

            # Process finger gesture
            finger_gesture_id = 0
            if len(pre_processed_point_history_list) == (self.history_length * 2):
                finger_gesture_id = self.point_history_classifier(
                    pre_processed_point_history_list)
            self.finger_gesture_history.append(finger_gesture_id)

            # Get most common gesture
            most_common_fg_id = Counter(self.finger_gesture_history).most_common()

            # Prepare results
            hand_result = {
                "gesture": self.keypoint_classifier_labels[hand_sign_id],
                "landmarks": landmark_list,
                "handedness": handedness.classification[0].label,
                "confidence": float(handedness.classification[0].score),
                "finger_gesture": self.point_history_classifier_labels[most_common_fg_id[0][0]]
                if most_common_fg_id else "none"
            }
            processed_results.append(hand_result)

            # Draw debug visualization
            debug_image = self.draw_landmarks(debug_image, landmark_list)
            brect = self.calc_bounding_rect(frame, hand_landmarks)
            debug_image = self.draw_bounding_rect(debug_image, brect)
            debug_image = self.draw_info_text(
                debug_image,
                brect,
                handedness,
                self.keypoint_classifier_labels[hand_sign_id],
                self.point_history_classifier_labels[most_common_fg_id[0][0]]
                if most_common_fg_id else "none"
            )

        return processed_results, debug_image

    # Helper methods from your original code
    def calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_point = []
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point.append([landmark_x, landmark_y])
        return landmark_point

    def pre_process_landmark(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]
            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))
        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))
        return temp_landmark_list

    # Include other helper methods (draw_landmarks, calc_bounding_rect, etc.)
    # from your original code...


class VideoProcessTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, track, processor):
        super().__init__()
        self.track = track
        self.processor = processor

    async def recv(self):
        frame = await self.track.recv()

        # Convert frame to CV2 format
        img = frame.to_ndarray(format="bgr24")

        # Process frame with hand gesture recognition
        results, debug_image = await self.processor.process_frame(img)

        # Convert processed frame back to MediaStreamTrack format
        new_frame = frame.from_ndarray(debug_image, format="bgr24")

        # Send results through WebSocket if available
        if hasattr(self, 'websocket') and results:
            try:
                await self.websocket.send(json.dumps({
                    "type": "gesture_data",
                    "data": results
                }))
            except Exception as e:
                print(f"Error sending results: {e}")

        return new_frame


class HandGestureServer:
    def __init__(self):
        self.pcs = set()
        self.processor = HandGestureProcessor()
        # Setup UDP socket
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.android_address = None  # Will be set when first message received

    async def handle_websocket(self, websocket):
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data["type"] == "offer":
                        pc = RTCPeerConnection()
                        self.pcs.add(pc)

                        @pc.on("track")
                        async def on_track(track):
                            if track.kind == "video":
                                video_processor = VideoProcessTrack(track, self.processor)

                                # Send processed data via UDP
                                async def send_processed_data(results):
                                    if self.android_address:
                                        for result in results:
                                            # Send landmarks
                                            landmarks_data = f"L:{json.dumps(result['landmarks'])}"
                                            self.udp_socket.sendto(landmarks_data.encode(),
                                                                   self.android_address)
                                            # Send gesture
                                            gesture_data = f"G:{result['gesture']}"
                                            self.udp_socket.sendto(gesture_data.encode(),
                                                                   self.android_address)

                                video_processor.on_processed_data = send_processed_data
                                pc.addTrack(video_processor)

                        await pc.setRemoteDescription(
                            RTCSessionDescription(sdp=data["sdp"], type="offer")
                        )

                        answer = await pc.createAnswer()
                        await pc.setLocalDescription(answer)

                        await websocket.send(json.dumps({
                            "type": "answer",
                            "sdp": pc.localDescription.sdp
                        }))

                except Exception as e:
                    print(f"Error handling message: {e}")

        finally:
            for pc in self.pcs:
                await pc.close()
            self.pcs.clear()

    async def start_udp_listener(self):
        # Listen for initial UDP message from Android to get its address
        self.udp_socket.bind(('0.0.0.0', 5052))
        while True:
            try:
                data, addr = self.udp_socket.recvfrom(1024)
                if data:
                    self.android_address = addr
                    print(f"Android device connected from: {addr}")
            except Exception as e:
                print(f"UDP Error: {e}")
                await asyncio.sleep(1)

    async def serve(self):
        # Start UDP listener in background
        asyncio.create_task(self.start_udp_listener())

        # Start WebSocket server
        async with serve(self.handle_websocket, "0.0.0.0", 8080):
            await asyncio.Future()  # run forever


if __name__ == "__main__":
    server = HandGestureServer()
    asyncio.run(server.serve())