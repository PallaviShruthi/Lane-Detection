"""
live_lane.py — YOLOv8 Lane Detection with Exact Polygon Tracing
================================================================
Works with:
  • Segmentation model (SEG) → traces exact lane polygon outlines from masks
  • Detection model     (DET) → draws perspective trapezoid overlays from boxes

Usage:
    python live_lane.py                            # runs on input.mp4 by default
    python live_lane.py --source input.mp4         # video file
    python live_lane.py --source 0                 # webcam
    python live_lane.py --source input.mp4 --save  # also saves output_lane_traced.mp4
    python live_lane.py --conf 0.25                # lower confidence = more detections
    python live_lane.py --model best.pt --source input.mp4

Controls (while window is open):
    Q / ESC  → quit
    S        → save screenshot
    F        → toggle filled overlay on/off
    C        → cycle lane colours  (neon → white → heat)
"""

import cv2
import numpy as np
import argparse
import time
import sys
from pathlib import Path
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────
# CLI Arguments
# ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='YOLOv8 Lane Detection')
parser.add_argument('--model',  default='best.pt',
                    help='Path to best.pt  (default: best.pt in current folder)')
parser.add_argument('--source', default='input.mp4',
                    help='Video path, image path, or 0 for webcam  (default: input.mp4)')
parser.add_argument('--conf',   type=float, default=0.25,
                    help='Confidence threshold 0-1  (default: 0.25)')
parser.add_argument('--imgsz',  type=int,   default=640,
                    help='Inference image size  (default: 640)')
parser.add_argument('--save',   action='store_true',
                    help='Save output video as output_lane_traced.mp4')
args = parser.parse_args()

# ──────────────────────────────────────────────────────────────
# Colour Palettes
# ──────────────────────────────────────────────────────────────
PALETTES = {
    'neon': [
        (0,   255,  80),   # bright green
        (0,   220, 255),   # cyan
        (255,  60, 200),   # magenta
        (255, 210,   0),   # yellow
        ( 80,  80, 255),   # blue
    ],
    'white': [
        (240, 240, 240),
        (200, 200, 200),
        (255, 255, 255),
        (180, 180, 180),
        (220, 220, 220),
    ],
    'heat': [
        (  0,   0, 255),   # red
        (  0, 128, 255),   # orange
        (  0, 255, 200),   # yellow-green
        (  0, 255,   0),   # green
        (255, 255,   0),   # cyan
    ],
}
PALETTE_NAMES = list(PALETTES.keys())
palette_idx   = 0   # global, changed by keypress


def get_colour(cls_id: int) -> tuple:
    pal = PALETTES[PALETTE_NAMES[palette_idx]]
    return pal[int(cls_id) % len(pal)]


# ──────────────────────────────────────────────────────────────
# Drawing — Segmentation model  (exact mask polygons)
# ──────────────────────────────────────────────────────────────

def draw_segmentation(frame: np.ndarray, masks, boxes, show_fill: bool) -> np.ndarray:
    """
    For each lane mask:
      1. Resize mask to frame size
      2. Find contour edges  →  these ARE the exact lane boundary lines
      3. Draw filled overlay (optional, semi-transparent)
      4. Draw thick outline  →  the lane trace
      5. Draw centre spine line fitted through the mask
    """
    H, W    = frame.shape[:2]
    overlay = frame.copy()

    for i, mask in enumerate(masks.data):
        cls_id = int(boxes.cls[i])
        conf   = float(boxes.conf[i])
        colour = get_colour(cls_id)

        # ── Resize mask to match frame ──
        m = mask.cpu().numpy().astype(np.uint8)
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)

        # ── Find polygon contours ──
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_TC89_L1)
        if not contours:
            continue

        # ── Filled overlay ──
        if show_fill:
            cv2.fillPoly(overlay, contours, colour)

        # ── Thick outline (the lane trace) ──
        cv2.polylines(frame, contours, isClosed=True,
                      color=colour, thickness=3, lineType=cv2.LINE_AA)

        # ── Centre spine line through each contour ──
        for cnt in contours:
            if cnt.shape[0] < 6:
                continue
            vx, vy, x0, y0 = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy, x0, y0 = float(vx), float(vy), float(x0), float(y0)
            length = max(W, H)
            pt1 = (np.clip(int(x0 - length * vx), 0, W-1),
                   np.clip(int(y0 - length * vy), 0, H-1))
            pt2 = (np.clip(int(x0 + length * vx), 0, W-1),
                   np.clip(int(y0 + length * vy), 0, H-1))
            cv2.line(frame, pt1, pt2, colour, 2, lineType=cv2.LINE_AA)

        # ── Label ──
        x1 = int(boxes.xyxy[i][0])
        y1 = int(boxes.xyxy[i][1])
        _draw_label(frame, f'Lane {conf:.2f}', x1, y1, colour)

    # ── Blend fill ──
    if show_fill:
        cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)

    return frame


# ──────────────────────────────────────────────────────────────
# Drawing — Detection model  (perspective trapezoid)
# ──────────────────────────────────────────────────────────────

def draw_detection(frame: np.ndarray, boxes, show_fill: bool) -> np.ndarray:
    """
    For each bounding box:
      1. Build a perspective trapezoid (tapers at top = vanishing point effect)
      2. Draw filled overlay (optional)
      3. Draw outline + centre spine line
    NOTE: For exact lane tracing, retrain with yolov8s-seg.pt (segmentation model)
    """
    H, W    = frame.shape[:2]
    overlay = frame.copy()

    for i in range(len(boxes.cls)):
        cls_id          = int(boxes.cls[i])
        conf            = float(boxes.conf[i])
        colour          = get_colour(cls_id)
        x1, y1, x2, y2 = map(int, boxes.xyxy[i])

        bw     = x2 - x1
        shrink = int(bw * 0.22)   # taper inward 22% at top

        poly = np.array([
            [x1,          y2],   # bottom-left
            [x2,          y2],   # bottom-right
            [x2 - shrink, y1],   # top-right  (tapered)
            [x1 + shrink, y1],   # top-left   (tapered)
        ], dtype=np.int32)

        if show_fill:
            cv2.fillPoly(overlay, [poly], colour)

        cv2.polylines(frame, [poly], isClosed=True,
                      color=colour, thickness=3, lineType=cv2.LINE_AA)

        # Centre spine
        cx_bot = (x1 + x2) // 2
        cx_top = cx_bot   # straight line for detection boxes
        cv2.line(frame, (cx_bot, y2), (cx_top, y1),
                 colour, 2, lineType=cv2.LINE_AA)

        _draw_label(frame, f'Lane {conf:.2f}', x1, y1, colour)

    if show_fill:
        cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)

    return frame


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _draw_label(frame: np.ndarray, text: str, x: int, y: int, colour: tuple):
    font        = cv2.FONT_HERSHEY_SIMPLEX
    scale, thk  = 0.55, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thk)
    lx = max(x, 0)
    ly = max(y - 4, th + 6)
    cv2.rectangle(frame, (lx, ly - th - 4), (lx + tw + 6, ly + 2), (0, 0, 0), -1)
    cv2.putText(frame, text, (lx + 3, ly), font, scale, colour, thk, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, fps: float, n_lanes: int,
             show_fill: bool, model_type: str, frame_no: int, total: int):
    H, W  = frame.shape[:2]
    font  = cv2.FONT_HERSHEY_SIMPLEX

    # ── Top bar ──
    cv2.rectangle(frame, (0, 0), (W, 36), (0, 0, 0), -1)
    cv2.putText(frame, f'FPS: {fps:5.1f}',  (  8, 24), font, 0.65, (  0, 255, 120), 1, cv2.LINE_AA)
    cv2.putText(frame, f'Lanes: {n_lanes}', (140, 24), font, 0.65, (  0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f'[{model_type}]',   (260, 24), font, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

    pal = PALETTE_NAMES[palette_idx].upper()
    cv2.putText(frame, f'Colour: {pal}', (W - 200, 24), font, 0.55, (200, 200,  60), 1, cv2.LINE_AA)

    fill_lbl = 'Fill: ON' if show_fill else 'Fill: OFF'
    cv2.putText(frame, fill_lbl, (W - 88, 24), font, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

    # ── Progress bar (video files only) ──
    if total > 0:
        prog = int(W * frame_no / total)
        cv2.rectangle(frame, (0, 36), (W,   40), ( 40,  40,  40), -1)
        cv2.rectangle(frame, (0, 36), (prog, 40), (  0, 200, 100), -1)

    # ── Bottom hint ──
    hint = 'Q/ESC=Quit   S=Screenshot   F=Fill   C=Colour'
    (tw, _), _ = cv2.getTextSize(hint, font, 0.42, 1)
    cv2.rectangle(frame, (0, H - 22), (W, H), (0, 0, 0), -1)
    cv2.putText(frame, hint, ((W - tw) // 2, H - 6),
                font, 0.42, (160, 160, 160), 1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    global palette_idx

    # ── Find model ──
    model_path = args.model
    if not Path(model_path).exists():
        candidates = sorted(Path('.').rglob('best.pt'))
        if candidates:
            model_path = str(candidates[0])
            print(f'[INFO] Model found at : {model_path}')
        else:
            print(f'[ERROR] Model not found: {args.model}')
            print('        Copy best.pt to this folder or use --model <path>')
            sys.exit(1)

    print(f'[INFO] Loading model  : {model_path}')
    model = YOLO(model_path)

    task       = getattr(model, 'task', 'detect')
    model_type = 'SEG' if task == 'segment' else 'DET'
    print(f'[INFO] Model type     : {model_type}')
    if model_type == 'DET':
        print('[WARN] DET model → bounding boxes only.')
        print('       For exact lane tracing → retrain using train_colab_seg.ipynb')

    # ── Open source ──
    src = args.source
    if isinstance(src, str) and src.isdigit():
        src = int(src)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f'[ERROR] Cannot open source: {args.source}')
        sys.exit(1)

    W_cap   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_cap   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cap = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   # 0 for webcam
    print(f'[INFO] Source         : {args.source}  ({W_cap}x{H_cap} @ {fps_cap:.1f} fps)')
    if total > 0:
        print(f'[INFO] Total frames   : {total}  (~{total/fps_cap:.1f}s)')

    # ── Optional video writer ──
    writer   = None
    out_path = 'output_lane_traced.mp4'
    if args.save or (isinstance(src, str) and Path(str(src)).is_file()):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps_cap, (W_cap, H_cap))
        print(f'[INFO] Saving output  : {out_path}')

    show_fill    = True
    screenshot_n = 0
    frame_no     = 0
    prev_time    = time.time()

    print('[INFO] Running — press Q or ESC to quit\n')

    while True:
        ret, frame = cap.read()
        if not ret:
            if total > 0:
                print('[INFO] End of video.')
                break
            continue   # webcam hiccup

        frame_no += 1

        # ── Inference ──
        results = model.predict(
            source  = frame,
            conf    = args.conf,
            imgsz   = args.imgsz,
            verbose = False,
        )[0]

        # ── Draw lanes ──
        if model_type == 'SEG' and results.masks is not None and len(results.masks.data):
            frame   = draw_segmentation(frame, results.masks, results.boxes, show_fill)
            n_lanes = len(results.masks.data)
        elif results.boxes is not None and len(results.boxes.cls):
            frame   = draw_detection(frame, results.boxes, show_fill)
            n_lanes = len(results.boxes.cls)
        else:
            n_lanes = 0

        # ── FPS ──
        now       = time.time()
        fps       = 1.0 / max(now - prev_time, 1e-9)
        prev_time = now

        # ── HUD ──
        draw_hud(frame, fps, n_lanes, show_fill, model_type, frame_no, total)

        # ── Display ──
        cv2.imshow('Lane Detection — Indian Campus Roads', frame)

        # ── Write frame ──
        if writer:
            writer.write(frame)

        # ── Key handling ──
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            print('[INFO] Quit by user.')
            break
        elif key == ord('f'):
            show_fill = not show_fill
            print(f'[INFO] Fill: {"ON" if show_fill else "OFF"}')
        elif key == ord('c'):
            palette_idx = (palette_idx + 1) % len(PALETTE_NAMES)
            print(f'[INFO] Colour palette: {PALETTE_NAMES[palette_idx]}')
        elif key == ord('s'):
            fname = f'screenshot_{screenshot_n:04d}.jpg'
            cv2.imwrite(fname, frame)
            print(f'[INFO] Screenshot saved: {fname}')
            screenshot_n += 1

    # ── Cleanup ──
    cap.release()
    if writer:
        writer.release()
        print(f'[INFO] Output saved: {out_path}')
    cv2.destroyAllWindows()
    print('[INFO] Done.')


if __name__ == '__main__':
    main()
    