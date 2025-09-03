import cv2
import numpy as np
import os
import time
import csv
from deepface import DeepFace

def load_images(image_folder):
    images = []
    class_names = []
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_folder, filename)
            images.append(img_path)
            class_names.append(os.path.splitext(filename)[0])
    return images, class_names

def encode_faces(image_paths):
    encodings = []
    valid_class_names = []
    
    for i, img_path in enumerate(image_paths):
        try:
            # Use DeepFace with strict detection for better quality encoding
            embedding = DeepFace.represent(
                img_path, 
                model_name="Facenet", 
                enforce_detection=True,  # Strict detection - no faces means error
                detector_backend="mtcnn"  # Use MTCNN for better face detection
            )
            if embedding:
                encodings.append(embedding[0]["embedding"])
                valid_class_names.append(os.path.splitext(os.path.basename(img_path))[0])
                print(f"Successfully encoded: {valid_class_names[-1]}")
        except Exception as e:
            print(f"Could not encode {img_path}: {str(e)}")
    
    return encodings, valid_class_names

def mark_attendance(name, attendance_data):
    with open('attendance.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, time.strftime("%Y-%m-%d %H:%M:%S")])
        print(f"Marked attendance for {name}")

def save_unknown_face(frame, face_location, confidence):
    """Save unknown face only if confidence is high enough and face is clear"""
    x, y, w, h = face_location
    
    # Higher thresholds for unknown face detection
    MIN_FACE_SIZE = 80  # Increased from 50
    MIN_CONFIDENCE = 0.4  # Higher threshold for unknown faces
    
    if w > MIN_FACE_SIZE and h > MIN_FACE_SIZE and confidence > MIN_CONFIDENCE:
        if not os.path.exists('unknown_faces'):
            os.makedirs('unknown_faces')
        
        face_image = frame[y:y+h, x:x+w]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join('unknown_faces', f'unknown_{timestamp}_conf_{confidence:.2f}.jpg')
        cv2.imwrite(file_path, face_image)
        print(f"Saved unknown face with confidence {confidence:.2f} to {file_path}")
        return True
    return False

def start_camera(image_folder, class_name, duration_in_seconds, recognition_threshold=0.65, unknown_threshold=0.4):
    image_paths, all_class_names = load_images(image_folder)
    encoded_faces, class_names = encode_faces(image_paths)
    
    if not encoded_faces:
        print("No faces could be encoded from the dataset. Please check your images.")
        return []

    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("Error: Could not open camera.")
        return []

    start_time = time.time()
    attendance_data = []
    recognized_names = set()  # Track recognized names for one-time marking
    known_class_names = set(class_names)

    # Use MTCNN for better face detection (more accurate than Haar cascades)
    print("Using MTCNN for face detection...")
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame.")
            break

        current_time = time.time()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            # Use DeepFace's built-in face detection with MTCNN (more accurate)
            face_objs = DeepFace.extract_faces(
                rgb_frame,
                detector_backend="mtcnn",  # More accurate face detection
                enforce_detection=False,
                align=True
            )
            
            for face_obj in face_objs:
                facial_area = face_obj["facial_area"]
                x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
                confidence = face_obj["confidence"]
                
                # Only process faces with good detection confidence
                if confidence < 0.9:  # High confidence threshold for detection
                    continue
                
                # Extract the face region
                face_img = frame[y:y+h, x:x+w]
                
                # Save temporary image for DeepFace processing
                temp_path = "temp_face.jpg"
                cv2.imwrite(temp_path, face_img)
                
                try:
                    # Get embedding for the detected face
                    face_embedding_obj = DeepFace.represent(
                        temp_path, 
                        model_name="Facenet", 
                        enforce_detection=False,
                        detector_backend="mtcnn"
                    )
                    
                    if face_embedding_obj:
                        face_embedding = face_embedding_obj[0]["embedding"]
                        name = "Unknown"
                        max_similarity = 0
                        
                        # Compare with known faces
                        for i, known_embedding in enumerate(encoded_faces):
                            similarity = np.dot(face_embedding, known_embedding) / (
                                np.linalg.norm(face_embedding) * np.linalg.norm(known_embedding)
                            )
                            
                            if similarity > max_similarity:
                                max_similarity = similarity
                                if similarity > recognition_threshold:
                                    name = class_names[i]
                        
                        # Mark attendance only once per person
                        if name != "Unknown":
                            if name not in recognized_names:
                                recognized_names.add(name)
                                attendance_data.append(name)
                                mark_attendance(name, attendance_data)
                                # Draw green rectangle for recognized faces
                                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                label = f"{name} ({max_similarity:.2f})"
                            else:
                                # Already recognized - draw blue rectangle
                                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                                label = f"{name} (Already marked)"
                        else:
                            # Check if this is a good quality unknown face
                            is_unknown_saved = save_unknown_face(frame, (x, y, w, h), max_similarity)
                            if is_unknown_saved:
                                # Draw yellow rectangle for saved unknown faces
                                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                            else:
                                # Draw red rectangle for low-quality detections
                                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                            label = f"Unknown ({max_similarity:.2f})"
                        
                        # Display label
                        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                except Exception as e:
                    print(f"Error processing face: {e}")
                
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        except Exception as e:
            print(f"Error in face detection: {e}")
            # Fallback to Haar cascades if MTCNN fails
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            face_locations = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        # Display timer
        elapsed_time = int(current_time - start_time)
        remaining_time = max(0, duration_in_seconds - elapsed_time)
        cv2.putText(frame, f"Time: {remaining_time}s | Recognized: {len(recognized_names)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Check for absentees if duration is completed
        if current_time - start_time > duration_in_seconds:
            for class_name in known_class_names:
                if class_name not in recognized_names:
                    mark_attendance(class_name + "_Absent", attendance_data)
            break

        cv2.imshow('Face Recognition Attendance System', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    return attendance_data

def run_attendance_system(image_folder, class_name, duration_in_seconds=120):
    print("Loading known faces...")
    attendance_data = start_camera(image_folder, class_name, duration_in_seconds)
    print("Attendance completed. Data:", attendance_data)
    return attendance_data

# Example usage
if __name__ == "__main__":
    run_attendance_system('dataset', 'Class_XII', duration_in_seconds=30)
