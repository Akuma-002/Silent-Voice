# main.py
from src.input.camera_handler import CameraHandler

def main():
    print("Starting Silent Voice...")
    cam = CameraHandler()
    cam.start_capture()

if __name__ == "__main__":
    main()
