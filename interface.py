import streamlit as st
import cv2
import mediapipe as mp
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.title("📸 AI Smart Selfie App")
st.write("Left dekho, Right dekho, aur Smile karo!")

if 'left_done' not in st.session_state:
    st.session_state.left_done = False
if 'right_done' not in st.session_state:
    st.session_state.right_done = False

st.sidebar.checkbox("Left Gesture", value=st.session_state.left_done)
st.sidebar.checkbox("Right Gesture", value=st.session_state.right_done)

class FaceProcessor(VideoTransformerBase):

    def __init__(self):
        self.prev_x = 0
        self.movement_threshold = 0.005
        self.stable_start_time = None
        self.photo_clicked = False
        self.cooldown = 5.0
        self.direction = {"Right": False, "Left": False}

        # ---- MediaPipe setup ONCE ----
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="face_landmarker.task"),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1
        )

        self.landmarker = FaceLandmarker.create_from_options(options)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp = int(time.time() * 1000)

        result = self.landmarker.detect_for_video(mp_image, timestamp)

        if result.face_landmarks:
            h, w, _ = img.shape
            nose = result.face_landmarks[0][4]
            curr_x, curr_y = nose.x, nose.y

            if self.prev_x != 0:
                x_diff = curr_x - self.prev_x
                diff = ((curr_x - self.prev_x)**2 +
                        (curr_y - curr_y)**2) ** 0.5

                # ---- Left / Right detection ----
                if x_diff > self.movement_threshold:
                    self.direction["Right"] = True
                    cv2.putText(img, "Right", (500, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

                elif x_diff < -self.movement_threshold:
                    self.direction["Left"] = True
                    cv2.putText(img, "Left", (50, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

                # ---- Stay still logic ----
                if diff < self.movement_threshold:
                    if self.stable_start_time is None:
                        self.stable_start_time = time.time()

                    elapsed = time.time() - self.stable_start_time

                    if (elapsed >= self.cooldown and
                        not self.photo_clicked and
                        self.direction["Right"] and
                        self.direction["Left"]):

                        cv2.imwrite(f"capture_{int(time.time())}.jpg", img)
                        self.photo_clicked = True
                        self.direction = {"Right": False, "Left": False}

                    text = ("SMILE!" if self.photo_clicked
                            else f"Stay Still: {int(self.cooldown-elapsed)+1}s")
                    cv2.putText(img, text, (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

                else:
                    self.stable_start_time = None
                    self.photo_clicked = False

            self.prev_x = curr_x

            # Draw landmarks
            for lm in result.face_landmarks[0]:
                x_px = int(lm.x * w)
                y_px = int(lm.y * h)
                cv2.circle(img, (x_px, y_px), 1, (0,0,255), -1)

        return img


webrtc_streamer(key="face-capture", video_transformer_factory=FaceProcessor)
