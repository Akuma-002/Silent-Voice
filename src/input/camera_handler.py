import cv2

class CameraHandler:
    def __init__(self, cam_index=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError("Camera not accessible")

    def read(self):
        ok, frame = self.cap.read()
        return ok, frame

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
