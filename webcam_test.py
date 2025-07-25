import cv2
import dlib
import torch
import numpy as np
from fusion_model import FusionModel
from temporal_model import TemporalConsistencyModel
from torchvision import transforms
import os
import bz2
import urllib.request
import time

# Download and extract landmarks file if needed
if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
    print("Downloading landmark file...")
    try:
        urllib.request.urlretrieve(
            "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
            "shape_predictor_68_face_landmarks.dat.bz2"
        )
        
        print("Extracting file...")
        with bz2.BZ2File("shape_predictor_68_face_landmarks.dat.bz2", "rb") as f_in:
            with open("shape_predictor_68_face_landmarks.dat", "wb") as f_out:
                f_out.write(f_in.read())
        
        print("Cleanup...")
        os.remove("shape_predictor_68_face_landmarks.dat.bz2")
    except Exception as e:
        print(f"Error downloading landmarks: {e}")
        print("Please download shape_predictor_68_face_landmarks.dat manually.")

# Initialize models
print("Loading fusion model...")
fusion_model = FusionModel()
fusion_model.load_state_dict(torch.load("results/best_fusion_model.pth"))
fusion_model.eval()

print("Loading temporal model...")
temporal_model = TemporalConsistencyModel()

# Load temporal model with filtered state_dict
try:
    print("Filtering and loading temporal model weights...")
    state_dict = torch.load("results/best_temporal_model.pth")
    
    # Filter out fusion_model keys
    filtered_dict = {k: v for k, v in state_dict.items() 
                    if not k.startswith("fusion_model.")}
    
    temporal_model.load_state_dict(filtered_dict, strict=False)
    temporal_model.eval()
    print("Temporal model loaded successfully!")
    use_temporal = True
except Exception as e:
    print(f"Error loading temporal model: {e}")
    print("Will use only fusion model")
    use_temporal = False

# Setup face detection
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Setup image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Emotion labels and settings
EMOTIONS = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust']
SEQ_LENGTH = 8
CONFIDENCE_THRESHOLD = 0.4  # Only show predictions with > 40% confidence

def main():
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Webcam opened successfully")
    prediction_buffer = []
    last_update = time.time()
    fps = 0
    frame_count = 0
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Process frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        # Calculate FPS
        frame_count += 1
        current_time = time.time()
        if current_time - last_update >= 1.0:
            fps = frame_count / (current_time - last_update)
            frame_count = 0
            last_update = current_time
        
        # Display FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        for face in faces:
            try:
                # Get landmarks
                landmarks = predictor(gray, face)
                landmarks_np = np.array([[p.x, p.y] for p in landmarks.parts()], dtype=np.float32)
                
                # CRUCIAL: Normalize landmarks exactly as during training
                landmarks_mean = np.mean(landmarks_np, axis=0)
                landmarks_std = np.std(landmarks_np)
                landmarks_np_normalized = (landmarks_np - landmarks_mean) / landmarks_std
                landmarks_tensor = torch.tensor(landmarks_np_normalized).unsqueeze(0).float()
                
                # Get face ROI
                x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                face_img = frame[y1:y2, x1:x2]
                
                # Skip if face is too small
                if face_img.size == 0 or face_img.shape[0] < 10 or face_img.shape[1] < 10:
                    continue
                    
                # Preprocess face image
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_tensor = transform(face_img).unsqueeze(0)
                
                # Fusion model prediction
                with torch.no_grad():
                    logits = fusion_model(face_tensor, landmarks_tensor)
                    probs = torch.softmax(logits, dim=1)[0]
                    fusion_pred = torch.argmax(probs).item()
                    fusion_conf = probs[fusion_pred].item()
                
                # Update temporal buffer
                prediction_buffer.append(probs.numpy())
                if len(prediction_buffer) > SEQ_LENGTH:
                    prediction_buffer.pop(0)
                
                # Temporal prediction
                if use_temporal and len(prediction_buffer) == SEQ_LENGTH:
                    seq_tensor = torch.tensor(np.array(prediction_buffer)).float().unsqueeze(0)
                    temporal_out = temporal_model(seq_tensor)
                    final_pred = torch.argmax(temporal_out, dim=1).item()
                    final_conf = torch.softmax(temporal_out, dim=1)[0, final_pred].item()
                else:
                    final_pred = fusion_pred
                    final_conf = fusion_conf
                
                # Apply confidence threshold for more reliable predictions
                emotion_text = EMOTIONS[final_pred]
                if final_conf < CONFIDENCE_THRESHOLD:
                    color = (0, 0, 255)  # Red = low confidence
                else:
                    color = (0, 255, 0)  # Green = high confidence
                
                # Draw landmarks
                for (x, y) in landmarks_np:
                    cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 255), -1)
                
                # Draw face bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Display predictions
                cv2.putText(frame, f"Emotion: {emotion_text}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"Conf: {final_conf:.2f}", (x1, y1 - 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Show fusion model prediction for comparison
                cv2.putText(frame, f"Frame: {EMOTIONS[fusion_pred]} ({fusion_conf:.2f})", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
            except Exception as e:
                print(f"Error processing face: {e}")
        
        # Display the frame
        cv2.imshow('Facial Expression Recognition', frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
