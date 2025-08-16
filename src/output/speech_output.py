import pyttsx3

class SpeechOutput:
    def __init__(self):
        self.engine = pyttsx3.init()

    def speak(self, text):
        print(f"[SPEAKING]: {text}")
        self.engine.say(text)
        self.engine.runAndWait()
