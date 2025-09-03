import cv2
import face_recognition
import numpy as np
import os
import time
import csv

def load_images(image_folder):
    images = []
    class_names = []
    for filename in os.listdir(image_folder):
        img_path = os.path.join(image_folder, filename)
        img = face_recognition.load_image_file(img_path)
        images.append(img)
        class_names.append(os.path.splitext(filename)[0])
    return images, class_names

def encode_faces(images):
    return [face_recognition.face_encodings(image)[0] for image in images]

def mark_attendance(name, attendance_data):
    with open('attendance.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, time.strftime("%Y-%m-%d %H:%M:%S")])
        print(f"Marked attendance for {name}")

def save_unknown_face(frame, face_location):
    # Create the 'unknown_faces' folder if it doesn't exist
    if not os.path.exists('unknown_faces'):
        os.makedirs('unknown_faces')
    
    top, right, bottom, left = face_location
    face_image = frame[top:bottom, left:right]  # Crop the face from the frame
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join('unknown_faces', f'unknown_{timestamp}.jpg')
    
    # Save the cropped face image
    cv2.imwrite(file_path, face_image)
    print(f"Saved unknown face to {file_path}")

def start_camera(image_folder, class_name, duration_in_seconds, threshold = 20):
    images, class_names = load_images(image_folder)
    encoded_faces = encode_faces(images)

    video_capture = cv2.VideoCapture(0)  # Start the camera

    start_time = time.time()
    attendance_data = []
    recognized_names = set()  # Set to keep track of recognized names

    # Create a set of known class names for attendance check
    known_class_names = set(class_names)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame.")
            break

        rgb_small_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)

        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(encoded_faces, face_encoding)
                name = "Unknown"

                if True in matches:
                    first_match_index = matches.index(True)
                    name = class_names[first_match_index]

                # Only mark attendance if the name is not already in recognized_names
                if name not in recognized_names:
                    recognized_names.add(name)  # Add name to recognized names
                    attendance_data.append(name)
                    mark_attendance(name, attendance_data)  # Mark attendance

                if name == "Unknown":
                    # Save the unknown face image
                    save_unknown_face(frame, face_location)

                # Optionally draw rectangles around detected faces
                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Check for absentees if duration is completed
        if time.time() - start_time > duration_in_seconds:
            for class_name in known_class_names:
                if class_name not in recognized_names:
                    mark_attendance(class_name + "_Absent", attendance_data)

            break

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    return attendance_data

def run_attendance_system(image_folder, class_name, duration_in_seconds=120):
    attendance_data = start_camera(image_folder, class_name, duration_in_seconds)
    print("Attendance data:", attendance_data)
    return attendance_data

# Example usage
run_attendance_system('dataset', 'Class_XII', duration_in_seconds=10)
