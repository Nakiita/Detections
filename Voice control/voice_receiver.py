import socket
from gpiozero import Robot
import RPi.GPIO as GPIO
from time import sleep
import cv2
import numpy as np
import threading

# Initialize GPIO and Robot control
GPIO.setmode(GPIO.BCM)
robot = Robot(left=(22, 23), right=(18, 27))

# Load class names from file
classNames = []
classFile = "/home/kritika/Downloads/Detections/Detections/coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# Configuration paths for the model
configPath = "/home/kritika/Downloads/Detections/Detections/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/kritika/Downloads/Detections/Detections/frozen_inference_graph.pb"

# Load the DNN model
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Global variable to track if a voice command was received
voice_command_received = False

def getObjects(img, thres, nms, objects=['stop sign', 'traffic light', 'car', 'truck']):
    """
    Detect and return objects with bounding boxes.
    """
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    objectInfo = []
    
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append(className)
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, f'{className.upper()} {confidence*100:.2f}%', (box[0], box[1]-10),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)

    return img, objectInfo

def handle_command(command):
    global voice_command_received

    voice_command_received = True

    if command == "stop":
        robot.stop()
        print("Stopping.")
    elif command == "forward":
        robot.forward()
        print("Moving forward.")
    elif command == "backward":
        robot.backward()
        print("Moving backward.")
    elif command == "left":
        robot.left()
        print("Turning left.")
    elif command == "right":
        robot.right()
        print("Turning right.")
    elif command == "dance":
        print("Making the vehicle dance.")
        for _ in range(4):
            robot.left()
            sleep(0.5)
            robot.right()
            sleep(0.5)
        robot.stop()
    else:
        print("Unknown command.")

    # Reset the flag to resume normal operation
    voice_command_received = False

def run_voice_command_listener():
    HOST = '0.0.0.0'
    PORT = 65432
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print(f'Server listening on {HOST}:{PORT}')

        conn, addr = server_socket.accept()
        print(f'Connected by {addr}')

        while True:
            command = conn.recv(1024).decode().strip()
            if not command:
                break

            print(f"Command received: {command}")
            handle_command(command)

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        conn.close()
        server_socket.close()
        print("Cleaned up resources.")

def run_object_detection():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    try:
        while True:
            success, img = cap.read()
            if not success:
                break

            if not voice_command_received:
                # Perform object detection
                img, detected_objects = getObjects(img, 0.45, 0.2)
                
                if 'stop sign' in detected_objects:
                    robot.stop()
                    print("Stopping, stop sign detected")
                elif 'traffic light' in detected_objects:
                    robot.right()
                    print("Turning right, light detected")
                    sleep(1.5)  # Adjust delay to complete the turn
                    robot.stop()
                else:
                    robot.forward()
                    print("Moving forward, no critical object detected")

            # Show the video feed with detection overlays
            cv2.imshow("Output", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by User")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Start the voice command listener in a separate thread
    voice_command_thread = threading.Thread(target=run_voice_command_listener)
    voice_command_thread.daemon = True
    voice_command_thread.start()

    # Run the object detection loop
    run_object_detection()

    # Clean up GPIO settings when done
    GPIO.cleanup()
