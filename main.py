import cv2
import face_recognition
import os
import numpy as np
import tkinter as tk
from tkinter import messagebox
from datetime import datetime
import csv

# Folder to store known faces
KNOWN_FACES_DIR = "known_faces"
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

# Load known faces
# Load known faces (Safe Version)
known_face_encodings = []
known_face_names = []

for filename in os.listdir(KNOWN_FACES_DIR):
    try:
        image_path = os.path.join(KNOWN_FACES_DIR, filename)

        # Load using OpenCV to validate format
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"Skipping {filename}: OpenCV couldn't read image.")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        encodings = face_recognition.face_encodings(img_rgb)
        if len(encodings) == 0:
            print(f"Skipping {filename}: No face found.")
            continue

        encoding = encodings[0]
        known_face_encodings.append(encoding)
        known_face_names.append(os.path.splitext(filename)[0])
        print(f"Loaded: {filename}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

# Attendance file
def mark_attendance(name):
    file_exists = os.path.exists("attendance.csv")
    date = datetime.now().strftime('%Y-%m-%d')
    time = datetime.now().strftime('%H:%M:%S')

    with open("attendance.csv", "a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Name", "Date", "Time"])
        writer.writerow([name, date, time])

# GUI functions
def start_recognition():
    video = cv2.VideoCapture(0)

    if not video.isOpened():
        messagebox.showerror("Error", "Could not access the webcam.")
        return

    while True:
        ret, frame = video.read()

        if not ret or frame is None:
            print("Error: Invalid frame received from webcam.")
            continue

        try:
            # Convert frame to RGB correctly
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if rgb_frame.dtype != np.uint8 or len(rgb_frame.shape) != 3:
                print("Error: Invalid RGB frame format.")
                continue

            # Face detection
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match = np.argmin(face_distances)
                    if matches[best_match]:
                        name = known_face_names[best_match]
                        mark_attendance(name)

                # Draw rectangle and label
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("Face Recognition", frame)

        except Exception as e:
            print(f"Error during recognition: {e}")
            continue

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

def show_help():
    messagebox.showinfo("Help", "1. Add images to 'known_faces' folder.\n"
                                "2. Name each image with person's name (e.g. samarth1.jpg).\n"
                                "3. Click 'Start Recognition' to begin.\n"
                                "4. Press 'q' in camera window to exit.")

# Tkinter GUI
root = tk.Tk()
root.title("Face Recognition Attendance System")
root.geometry("400x300")

label = tk.Label(root, text="Face Recognition Attendance", font=("Helvetica", 16))
label.pack(pady=20)

btn_start = tk.Button(root, text="Start Recognition", command=start_recognition, height=2, width=25)
btn_start.pack(pady=10)

btn_help = tk.Button(root, text="How to Use", command=show_help, height=2, width=25)
btn_help.pack(pady=10)

btn_exit = tk.Button(root, text="Exit", command=root.quit, height=2, width=25)
btn_exit.pack(pady=10)

root.mainloop()