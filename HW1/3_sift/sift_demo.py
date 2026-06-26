import sys
import time


import cv2
import numpy as np


# Constants

_rng_colors  = np.random.default_rng(42)
_MATCH_COLORS = [tuple(int(v) for v in _rng_colors.integers(80, 255, size=3))
                 for _ in range(500)]

KP_COLOR = (0, 210, 0)
HEADER_H = 75


# Helpers

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def to_gray(img):
    return _clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

def resize_keep_aspect(img, max_h=480, max_w=640):
    h, w = img.shape[:2]
    s = min(max_h / h, max_w / w)
    if s < 1:
        return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


# Visualization

def draw_keypoints_on_canvas(canvas, keypoints, offset_x=0, offset_y=0):
    for kp in keypoints:
        x = int(kp.pt[0]) + offset_x
        y = int(kp.pt[1]) + offset_y
        r = max(int(kp.size / 2), 3)
        angle_rad = np.deg2rad(kp.angle)
        ex = int(x + r * np.cos(angle_rad))
        ey = int(y + r * np.sin(angle_rad))
        cv2.circle(canvas, (x, y), r, KP_COLOR, 1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 2, KP_COLOR, -1, cv2.LINE_AA)
        cv2.line(canvas,   (x, y), (ex, ey), KP_COLOR, 1, cv2.LINE_AA)


def build_matching_frame(img_ref, kp_ref, img_live, kp_live, matches,
                         ratio_thresh, fps):
    h1, w1 = img_ref.shape[:2]
    h2, w2 = img_live.shape[:2]
    GAP     = 8
    H       = max(h1, h2) + HEADER_H
    W       = w1 + GAP + w2
    offset2 = w1 + GAP
    y0      = HEADER_H

    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    # header
    cv2.rectangle(canvas, (0, 0), (W, HEADER_H), (30, 30, 30), -1)
    line1 = (f"Matches: {len(matches)}   "
             f"KP ref/live: {len(kp_ref)}/{len(kp_live)}   "
             f"Ratio: {ratio_thresh:.2f}   FPS: {fps:.1f}")
    line2 = "SPACE=Recapture   -= stricter(fewer)   += looser(more)   Q=Quit"
    cv2.putText(canvas, line1, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, line2, (10, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 60), 1, cv2.LINE_AA)

    # images
    canvas[y0:y0 + h1, :w1]            = img_ref
    canvas[y0:y0 + h2, offset2:offset2 + w2] = img_live
    canvas[y0:, w1:w1 + GAP]           = (55, 55, 55)

    # keypoints
    draw_keypoints_on_canvas(canvas, kp_ref,  offset_x=0,       offset_y=y0)
    draw_keypoints_on_canvas(canvas, kp_live, offset_x=offset2, offset_y=y0)

    # match lines
    sorted_m = sorted(matches, key=lambda m: m.distance)[:80]
    for i, m in enumerate(sorted_m):
        pt1 = (int(kp_ref[m.queryIdx].pt[0]),
               int(kp_ref[m.queryIdx].pt[1])  + y0)
        pt2 = (int(kp_live[m.trainIdx].pt[0]) + offset2,
               int(kp_live[m.trainIdx].pt[1]) + y0)
        cv2.line(canvas, pt1, pt2, _MATCH_COLORS[i % len(_MATCH_COLORS)], 2, cv2.LINE_AA)

    # labels
    label_y = y0 + max(h1, h2) - 6
    cv2.putText(canvas, "Reference (captured)", (6, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Live", (offset2 + 6, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    return canvas


def build_preview_frame(img_live, fps):
    """Full-screen live feed with instruction overlay."""
    frame = img_live.copy()
    h, w  = frame.shape[:2]
    # dim overlay bar at bottom
    cv2.rectangle(frame, (0, h - 60), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(frame, 0.6, img_live, 0.4, 0, frame)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, "Press SPACE to capture reference image",
                (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Q=Quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 60), 1, cv2.LINE_AA)
    return frame


def build_idle_frame(w=1920, h=540):
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(canvas, "Press SPACE to open camera",
                (w // 2 - 280, h // 2 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 230, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Q = Quit",
                (w // 2 - 70, h // 2 + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 60), 1, cv2.LINE_AA)
    return canvas


# Main

def main():
    sift = cv2.SIFT_create(
        nfeatures=800,
        nOctaveLayers=4,
        contrastThreshold=0.03,
        edgeThreshold=12,
    )
    bf = cv2.BFMatcher(cv2.NORM_L2)

    # states: 'idle' → 'preview' → 'matching'
    state        = 'idle'
    cap          = None
    img_ref      = None
    kp_ref       = None
    des_ref      = None
    ratio_thresh = 0.75
    t_prev       = time.time()

    cv2.namedWindow("SIFT Camera Demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SIFT Camera Demo", 1920, 540)
    cv2.imshow("SIFT Camera Demo", build_idle_frame())

    print("SIFT Camera Demo")
    print(">>> Click the OpenCV window to focus, then press keys <<<")
    print("  SPACE=Open camera/Capture   -/+=Threshold   Q=Quit")

    while True:
        now = time.time()
        fps = 1.0 / max(now - t_prev, 1e-6)
        t_prev = now

        # grab live frame
        live = None
        if state in ('preview', 'matching') and cap is not None:
            ret, raw = cap.read()
            if ret:
                live = resize_keep_aspect(raw, max_h=720, max_w=960)

        # build display
        if state == 'idle':
            frame = build_idle_frame()

        elif state == 'preview':
            if live is not None:
                frame = build_preview_frame(live, fps)
            else:
                frame = build_idle_frame()

        elif state == 'matching':
            if live is not None and des_ref is not None:
                kp_live, des_live = sift.detectAndCompute(to_gray(live), None)
                good = []
                if des_live is not None and len(des_live) >= 2:
                    pairs = bf.knnMatch(des_ref, des_live, k=2)
                    for pair in pairs:
                        if len(pair) == 2 and pair[0].distance < ratio_thresh * pair[1].distance:
                            good.append(pair[0])
                frame = build_matching_frame(img_ref, kp_ref, live, kp_live,
                                             good, ratio_thresh, fps)
            else:
                frame = build_idle_frame()

        cv2.imshow("SIFT Camera Demo", frame)

        # keyboard
        key = cv2.waitKey(1)
        if key == -1:
            continue
        ascii_key = key & 0xFF

        if ascii_key in (ord('q'), 27):
            break

        elif ascii_key == ord(' '):
            if state == 'idle':
                # on macOS use AVFoundation backend to trigger permission dialog
                backend = cv2.CAP_AVFOUNDATION if sys.platform == 'darwin' else cv2.CAP_ANY
                cap = cv2.VideoCapture(0, backend)
                if not cap.isOpened():
                    print("Cannot open camera.")
                    print("macOS: System Settings → Privacy & Security → Camera")
                    print("       → Grant camera permission to this terminal (or VS Code).")
                    print("       Restart the program after granting permission.")
                else:
                    state = 'preview'
                    print("Camera opened. Press SPACE to capture the reference image.")

            elif state == 'preview' and live is not None:
                # capture reference
                img_ref = live.copy()
                kp_ref, des_ref = sift.detectAndCompute(to_gray(img_ref), None)
                state = 'matching'
                print(f"Capture complete. keypoints: {len(kp_ref)}  (SPACE=Recapture)")

            elif state == 'matching':
                # recapture: go back to preview
                img_ref = None; kp_ref = None; des_ref = None
                state = 'preview'
                print("Recapture mode. Press SPACE to capture a new reference.")

        elif ascii_key == ord('-'):
            ratio_thresh = round(max(0.40, ratio_thresh - 0.05), 2)
            print(f"ratio_thresh → {ratio_thresh}")
        elif ascii_key in (ord('+'), ord('=')):
            ratio_thresh = round(min(0.95, ratio_thresh + 0.05), 2)
            print(f"ratio_thresh → {ratio_thresh}")

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
