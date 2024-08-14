import socket
import pyttsx3
import speech_recognition as sr
import spacy

# Initialize pyttsx3 for Text-to-Speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)

# Load Spacy model
nlp = spacy.load('en_core_web_sm')

# Client setup
HOST = '192.168.137.211'  # Replace with the IP address of your Raspberry Pi
PORT = 65432              # The port used by the server

def speak(text):
    engine.say(text)
    engine.runAndWait()

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for commands...")
        audio = recognizer.listen(source)
        try:
            command = recognizer.recognize_google(audio, language='en-US').lower()
            print(f"Command received: {command}")
            return command
        except sr.UnknownValueError:
            print("Could not understand the command.")
            speak("Sorry, I did not understand the command.")
            return None
        except sr.RequestError:
            print("Speech Recognition service is not available.")
            speak("Sorry, the speech recognition service is not available.")
            return None

def lemmatize_command(command):
    doc = nlp(command)
    lemmas = [token.lemma_ for token in doc]
    return ' '.join(lemmas), doc

def handle_command(command, s):
    command_map = {
        'stop': ["stop"],
        'forward': ["forward", "go", "move", "keep going"],
        'backward': ["backward", "reverse", "back"],
        'left': ["left", "turn left"],
        'right': ["right", "turn right"],
        'dance': ["dance"]
    }

    vehicle_commands = {
        'stop': "Stopping the vehicle.",
        'forward': "Moving the vehicle forward.",
        'backward': "Moving the vehicle backward.",
        'left': "Turning the vehicle left.",
        'right': "Turning the vehicle right.",
        'dance': "Making the vehicle dance."
    }

    for action, keywords in command_map.items():
        if any(keyword in command for keyword in keywords):
            s.sendall(action.encode())
            speak(vehicle_commands[action])
            return
    
    # General interactions
    if 'hello' in command:
        speak("Hello, how can I help you?")
    elif 'how are you' in command:
        speak("I am fine, thank you. How can I assist you today?")
    elif 'what is your name' in command:
        speak("I am your voice assistant.")
    else:
        speak("I am sorry, I don't understand that command.")

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    try:
        s.connect((HOST, PORT))
        while True:
            command = recognize_speech()
            if command:
                command, doc = lemmatize_command(command)
                handle_command(command, s)
    except Exception as e:
        print(f"An error occurred: {e}")
        speak("An error occurred. Please check the connection and try again.")
