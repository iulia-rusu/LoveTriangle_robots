import cv2
import glob
import os

TARGET_SECONDS = 30.0
HERE = os.path.dirname(__file__)

for path in sorted(glob.glob(os.path.join(HERE, "*.avi"))):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    old_dur = n_frames / fps if fps else 0

    new_fps = n_frames / TARGET_SECONDS
    tmp_path = path + ".tmp.avi"
    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"FMP4"), new_fps, (width, height))

    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
    cap.release()
    writer.release()

    os.replace(tmp_path, path)
    print(f"{os.path.basename(path):55s} {old_dur:6.2f}s -> 30.00s  ({written} frames @ {new_fps:.2f}fps)")
