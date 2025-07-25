import cv2
import dlib
import os
import numpy as np
from imutils import face_utils

# Load dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Place in the same folder

# Set your folders
input_dir = "VideoFlash"
output_dir = "output_landmarks"
os.makedirs(output_dir, exist_ok=True)

def extract_landmarks_from_video(video_path, save_folder):
    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    basename = os.path.splitext(os.path.basename(video_path))[0]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if faces:
            for face in faces:
                shape = predictor(gray, face)
                landmarks = face_utils.shape_to_np(shape)  # (68, 2)
                np.save(os.path.join(save_folder, f"{basename}_frame_{frame_num:03d}.npy"), landmarks)
                break  # only process the first detected face
        frame_num += 1

    cap.release()

# Process each video in the folder
for video_file in os.listdir(input_dir):
    if video_file.endswith(".mp4"):
        video_path = os.path.join(input_dir, video_file)
        extract_landmarks_from_video(video_path, output_dir)
        print(f"[✔] Processed: {video_file}")
