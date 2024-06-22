import cv2
import os

# Directory to save captured face images
capture_dir = "known_faces/Nakita"
os.makedirs(capture_dir, exist_ok=True)

# Initialize the camera
video_capture = cv2.VideoCapture(0)

# Face detection parameters
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
min_face_size = (30, 30)
scale_factor = 1.1
min_neighbors = 5

# Counter for captured images
image_counter = 0

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors,
                                          minSize=min_face_size)

    # Draw rectangles around the faces and capture images
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Save the captured face image
        image_path = os.path.join(capture_dir, f"image{image_counter}.jpg")
        cv2.imwrite(image_path, frame[y:y+h, x:x+w])
        image_counter += 1

    # Display the resulting frame
    cv2.imshow('Capture Faces', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q') or image_counter >= 10:
        break

# Release the capture
video_capture.release()
cv2.destroyAllWindows()
