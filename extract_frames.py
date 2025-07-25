import cv2
import os
import re

# Settings
VIDEO_DIR = "VideoFlash"
OUTPUT_IMAGE_DIR = "output_images"
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

# Process all videos
for video_file in os.listdir(VIDEO_DIR):
    if video_file.endswith(".mp4"):
        video_path = os.path.join(VIDEO_DIR, video_file)
        basename = os.path.splitext(video_file)[0]
        
        print(f"Processing video: {video_file}")
        cap = cv2.VideoCapture(video_path)
        frame_num = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save with same naming pattern as your landmark files
            img_name = f"{basename}_frame_{frame_num:03d}.jpg"
            output_path = os.path.join(OUTPUT_IMAGE_DIR, img_name)
            cv2.imwrite(output_path, frame)
            frame_num += 1
            
            if frame_num % 100 == 0:
                print(f"  - Extracted {frame_num} frames")
                
        cap.release()
        print(f"  ✓ Completed: {frame_num} frames extracted")

print(f"Extraction complete! Images saved to {OUTPUT_IMAGE_DIR}")
