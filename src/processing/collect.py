import argparse, csv, os, time
import cv2
from pathlib import Path
from collections import deque
from src.input.camera_handler import CameraHandler
from src.utils.landmarks import HandLandmarker

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="gesture label, e.g. hello")
    ap.add_argument("--num", type=int, default=300, help="samples to collect")
    ap.add_argument("--outfile", default="data/gestures.csv")
    ap.add_argument("--cam", type=int, default=0)
    args = ap.parse_args()

    Path("data").mkdir(parents=True, exist_ok=True)
    file_exists = os.path.isfile(args.outfile)

    cam = CameraHandler(cam_index=args.cam)
    hlm = HandLandmarker()
    collected = 0
    recent_detect = deque(maxlen=10)

    with open(args.outfile, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["f"+str(i) for i in range(21*3)] + ["label"])  # 63 + label

        print(f"[INFO] Hold the gesture '{args.label}' in front of the camera.")
        print("[INFO] Press 's' to start capturing, 'q' to quit.")
        started = False

        while True:
            ok, frame = cam.read()
            if not ok: break

            results = hlm.process(frame)

            if results.multi_hand_landmarks:
                for idx, hl in enumerate(results.multi_hand_landmarks):
                    if results.multi_handedness:
                        handed = results.multi_handedness[idx].classification[0].label
                    else:
                        handed = None

                    feats = HandLandmarker.to_features(hl, include_z=True, include_handedness=False)

                    # simple overlay
                    cv2.putText(frame, f"Label: {args.label}  Collected: {collected}/{args.num}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('s'):
                        started = True
                        print("[INFO] capture started")
                    if key == ord('q'):
                        cam.release()
                        return

                    if started and collected < args.num:
                        writer.writerow(list(feats) + [args.label])
                        collected += 1

            cv2.imshow("Collect - Silent Voice", frame)
            if collected >= args.num:
                print("[INFO] Done.")
                time.sleep(0.5)
                break

    cam.release()

if __name__ == "__main__":
    main()
