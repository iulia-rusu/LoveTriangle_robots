import cv2
import glob
import os

for path in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "*.avi"))):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    dur = n / fps if fps else 0
    print(f"{os.path.basename(path):55s} fps={fps:6.2f} frames={n:7.0f} dur={dur:7.2f}s fourcc={fourcc_str!r}")
    cap.release()
