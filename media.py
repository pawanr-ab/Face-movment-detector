import cv2
import mediapipe as mp
import time

# 1. Setup - Shortcuts aur Options
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = 'face_landmarker.task' 

# Video mode select karein
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

# 2. Camera start karein
cap = cv2.VideoCapture(0)
prev_x = 0
prev_y=0
movement_threshold = 0.005
stable_start_time = None
photo_clicked = False
cooldown = 5.0
direction={"Right":False, "Left":False}

# 3. Landmarker ko Context Manager (with) ke saath open karein
with FaceLandmarker.create_from_options(options) as landmarker:
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Frame ko flip karein (mirror effect ke liye)
        frame = cv2.flip(frame, 1)

        # OpenCV BGR deta hai, MediaPipe RGB mangta hai
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipe Image object banayein
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # VIDEO mode mein 'timestamp' dena compulsory hai (milliseconds mein)
        frame_timestamp_ms = int(time.time() * 1000)

        # Detection chalayein
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        # --- Drawing Logic ---
        if result.face_landmarks:
            for face_landmarks in result.face_landmarks:
                # Sirf nose aur aankhon ke points draw karne ka example
                h, w, _ = frame.shape
                nose = face_landmarks[4]
                curr_x,curr_y = nose.x,nose.y
                if prev_x != 0: # First frame skip karne ke liye
                    diff = ((curr_x - prev_x)**2 + (curr_y - prev_y)**2)**0.5
                    x_diff = curr_x - prev_x

                    # --- DIRECTION LOGIC ---
                    if x_diff > movement_threshold:
                        direction['Right'] = True
                        cv2.putText(frame, "Right", (500, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                    elif x_diff < -movement_threshold:
                        direction['Left'] = True
                        cv2.putText(frame, "Left", (50,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                    #euclidian formula
                    if diff < movement_threshold:
                    # FACE STABLE HAI

                        if stable_start_time is None:
                            stable_start_time = time.time()
                        
                        elapsed = time.time() - stable_start_time
                        
                        if elapsed >= cooldown and not photo_clicked and direction["Right"] and direction["Left"]:
                            cv2.imwrite(f"capture_{int(time.time())}.jpg", frame)
                            print("PHOTO CLICKED!")
                            photo_clicked = True
                            direction.update({"Right": False, "Left": False})
                        if photo_clicked:
                            cv2.putText(frame, "PHOTO CAPTURED!", (w//2-100, h-50), 
                                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
                        # UI par countdown dikhayein
                        color = (0, 255, 0) if photo_clicked else (0, 255, 255)
                        text = "SMILE!" if photo_clicked else f"Stay Still: {int(cooldown-elapsed)+1}s"
                        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        
                    else:
                        # FACE MOVE HUA - RESET
                        stable_start_time = None
                        photo_clicked = False
                print(direction)
                prev_x, prev_y = curr_x, curr_y
                for landmark in face_landmarks:
                    x_px = int(landmark.x * w)
                    y_px = int(landmark.y * h)
                    cv2.circle(frame, (x_px, y_px), 1, (0, 0, 255), -1)

        # Result dikhayein
        cv2.imshow('MediaPipe Face Landmarker', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()