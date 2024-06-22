import cv2
import face_recognition
import os

# Load known faces and encode them
known_faces_dir = "known_faces"
known_faces_encodings = []
known_faces_names = []

# Iterate through each subdirectory (each person's directory) in the known_faces directory
for person_dir in os.listdir(known_faces_dir):
    person_name = person_dir.capitalize()
    person_path = os.path.join(known_faces_dir, person_dir)

    # Iterate through each image file in the person's directory
    for image_file in os.listdir(person_path):
        image_path = os.path.join(person_path, image_file)
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)

        # Check if a face was detected in the image
        if len(face_encodings) > 0:
            # Append the first face encoding and name to the respective lists
            face_encoding = face_encodings[0]
            known_faces_encodings.append(face_encoding)
            known_faces_names.append(person_name)

# Initialize the camera
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Find faces in the frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Check if any face is found in the frame
    if len(face_encodings) > 0:
        # Iterate through each face found in the frame
        for face_encoding in face_encodings:
            # Compare the face encoding with all known face encodings
            matches = face_recognition.compare_faces(known_faces_encodings, face_encoding)
            name = "Unknown"

            # Check if any known face matches the current face
            if True in matches:
                # Find the index of the matched face
                matched_index = matches.index(True)
                name = known_faces_names[matched_index]

            # Draw a label with the name on the frame
            top, right, bottom, left = face_locations[0]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
video_capture.release()
cv2.destroyAllWindows()
