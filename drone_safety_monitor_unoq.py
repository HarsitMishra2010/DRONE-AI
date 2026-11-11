"""
Drone Safety Monitor v5 - tuned starting configuration for Arduino UNO Q 4GB.

Computer vision runs on the UNO Q's Debian/Linux MPU (QRB2210). The separate
STM32 microcontroller can be added later for LEDs, buzzer, and sensor I/O.

Conservative starting settings: YOLOv8n-pose, 416 input, 480px frame width,
CPU inference, one inference on every two captured frames, and limited detections.
Actual speed depends on the camera, lighting, software builds, and model runtime.

Examples:
    python drone_safety_monitor_unoq.py --source 0
    python drone_safety_monitor_unoq.py --source 0 --show-pose
    python drone_safety_monitor_unoq.py --source sample.mp4 --skip-frames 0

Important: this script requires ARM64-compatible Python packages, including a
working PyTorch build supported by the installed Ultralytics version. If that
stack is unavailable on your UNO Q image, use an exported ONNX model with an
ARM64-compatible ONNX Runtime/OpenCV DNN implementation instead.

Alerts are prompts for human review, not proof that an incident occurred.
"""

import os
# Keep numerical libraries from creating too many worker threads on a small SBC.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

#this segment imports all the necessary libraries such as argparse,cv2,ultralytics,etc.
import argparse
import time
from collections import defaultdict, deque
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

#this segment tries to import torch and winsound
#for torch it tries to import the two types those being set_num_threads(2) and set_num_interop_threads(1)
try:
    import torch
    torch.set_num_threads(2)        
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
except ImportError:
    torch = None

try:
    import winsound  # Windows only
except ImportError:
    winsound = None


KP_CONF = 0.50
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12


def is_lying(box, kpts, kp_conf=KP_CONF):
    """
    Estimate whether a person has a horizontal posture.
    Returns (lying, diagnostic_text). This is a visual estimate, not a diagnosis.
    """
    x1, y1, x2, y2 = box
    width = max(float(x2 - x1), 1.0)
    height = max(float(y2 - y1), 1.0)
    aspect = width / height

    shoulders = [
        kpts[i, :2] for i in (LEFT_SHOULDER, RIGHT_SHOULDER)
        if len(kpts[i]) >= 3 and kpts[i, 2] >= kp_conf
    ]
    hips = [
        kpts[i, :2] for i in (LEFT_HIP, RIGHT_HIP)
        if len(kpts[i]) >= 3 and kpts[i, 2] >= kp_conf
    ]

    if shoulders and hips:
        shoulder_center = np.mean(shoulders, axis=0)
        hip_center = np.mean(hips, axis=0)
        dx = abs(float(shoulder_center[0] - hip_center[0]))
        dy = abs(float(shoulder_center[1] - hip_center[1]))
        angle = float(np.degrees(np.arctan2(dx, max(dy, 1e-6))))
        lying = angle >= 55.0 and aspect >= 0.85
        return lying, f"torso {angle:.0f}deg | box w/h {aspect:.2f}"

    # If the torso keypoints are hidden, use a stricter box-only fallback.
    lying = aspect >= 1.55
    return lying, f"torso uncertain | box w/h {aspect:.2f}"


def estimate_drop(history, now, current_y, current_size):
    """Return the recent downward displacement in body-size units, if measurable."""
    candidates = [
        item for item in history
        if now - item[0] >= 0.25 and now - item[0] <= 1.25
    ]
    if not candidates:
        return None

    # Use the oldest suitable observation to estimate the recent change.
    old_time, old_y, old_size = candidates[0]
    scale = max(float(old_size), float(current_size), 1.0)
    return (float(current_y) - float(old_y)) / scale


def find_groups(centers, sizes, min_size, dist_factor):
    """Find connected groups using normalized image-space distance."""
    n = len(centers)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            limit = dist_factor * max(sizes[i], sizes[j])
            if np.linalg.norm(centers[i] - centers[j]) < limit:
                parent[find(i)] = find(j)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return [group for group in groups.values() if len(group) >= min_size]


def draw_label(frame, text, x, y, color):
    y = max(int(y), 18)
    cv2.putText(
        frame, text, (int(x), y), cv2.FONT_HERSHEY_SIMPLEX,
        0.52, color, 2, cv2.LINE_AA
    )


def run(args):
    source = int(args.source) if args.source.isdigit() else args.source
    # Limit OpenCV's own worker pool and ask live cameras for a short buffer.
    cv2.setNumThreads(2)
    cap = cv2.VideoCapture(source)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    print("Loading pose model. First run may download the model file.")
    model = YOLO(args.model, task="pose")
    os.makedirs(args.alert_dir, exist_ok=True)
    print("UNO Q profile:", args.model, "| device:", args.device,
          "| imgsz:", args.imgsz, "| width:", args.width,
          "| skip_frames:", args.skip_frames)

    # Per-track state
    posture_since = {}
    lying_since = {}
    fall_alerted = set()
    person_down_alerted = set()
    last_seen = {}
    y_hist = defaultdict(deque)
    prev_rel = {}
    arm_speed = defaultdict(float)

    # Pair state for cautious review alerts
    interaction_since = {}
    interaction_alerted = set()
    last_group_log = 0.0

    frame_times = deque(maxlen=60)
    last_stats_print = 0.0
    frame_count = 0
    last_people_count = 0
    last_inference_ms = 0.0

    def trigger(kind, message, frame):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_kind = "".join(c for c in kind if c.isalnum() or c in ("_", "-"))
        image_path = os.path.join(args.alert_dir, f"{safe_kind}_{stamp}.jpg")
        saved = cv2.imwrite(image_path, frame)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] REVIEW ALERT: {message}")
        if saved:
            print(f"  Snapshot: {image_path}")
        if winsound is not None and not args.no_beep:
            try:
                winsound.PlaySound(
                    "SystemExclamation",
                    winsound.SND_ALIAS | winsound.SND_ASYNC
                )
            except RuntimeError:
                pass

    try:
        while True:
            ok = cap.grab()
            if not ok:
                break

            frame_count += 1
            if args.skip_frames and frame_count % (args.skip_frames + 1) != 0:
                continue

            ok, frame = cap.retrieve()
            if not ok:
                break

            if args.width > 0 and frame.shape[1] != args.width:
                scale = args.width / frame.shape[1]
                frame = cv2.resize(
                    frame, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
                )

            now = time.time()
            frame_times.append(now)

            # Optional warm-up is handled by Ultralytics on first inference.
            track_kwargs = {
                "persist": True,
                "tracker": args.tracker,
                "imgsz": args.imgsz,
                "conf": args.conf,
                "iou": args.iou,
                "max_det": args.max_det,
                "verbose": False,
            }
            track_kwargs["device"] = args.device
            # FP16 is deliberately disabled in the UNO Q CPU profile.
            if args.half and args.device.lower() != "cpu":
                track_kwargs["half"] = True

            inference_start = time.perf_counter()
            result = model.track(frame, **track_kwargs)[0]
            last_inference_ms = (time.perf_counter() - inference_start) * 1000

            # Draw pose keypoints/skeleton first; custom boxes and labels go on top.
            if args.show_pose:
                try:
                    frame = result.plot(boxes=False, labels=False, kpt_line=True)
                except TypeError:
                    # Compatibility with older Ultralytics versions.
                    frame = result.plot(boxes=False, labels=False)

            ids, boxes_i, centers, sizes = [], [], [], []
            people_count = 0

            if (
                result.boxes is not None
                and result.boxes.id is not None
                and result.keypoints is not None
                and result.keypoints.data is not None
            ):
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().tolist()
                kpts_all = result.keypoints.data.cpu().numpy()

                for box, tid, kpts in zip(boxes, track_ids, kpts_all):
                    x1, y1, x2, y2 = box.astype(int)
                    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=float)
                    size = max(float(x2 - x1), float(y2 - y1), 1.0)

                    ids.append(tid)
                    boxes_i.append((x1, y1, x2, y2))
                    centers.append(center)
                    sizes.append(size)
                    last_seen[tid] = now
                    people_count += 1

                    hist = y_hist[tid]
                    hist.append((now, center[1], size))
                    while hist and now - hist[0][0] > 1.5:
                        hist.popleft()

                    # Smooth wrist movement relative to the body centre.
                    speed = 0.0
                    if kpts.shape[0] > max(LEFT_WRIST, RIGHT_WRIST):
                        wrists = kpts[[LEFT_WRIST, RIGHT_WRIST]]
                        rel = np.column_stack([
                            wrists[:, :2] - center,
                            wrists[:, 2]
                        ])
                        if tid in prev_rel:
                            prev_t, previous = prev_rel[tid]
                            dt = max(now - prev_t, 1e-3)
                            for wrist_index in range(2):
                                if (
                                    rel[wrist_index, 2] >= KP_CONF
                                    and previous[wrist_index, 2] >= KP_CONF
                                ):
                                    distance = np.linalg.norm(
                                        rel[wrist_index, :2] -
                                        previous[wrist_index, :2]
                                    )
                                    speed = max(speed, distance / dt / size)
                        prev_rel[tid] = (now, rel)
                    arm_speed[tid] = (
                        0.65 * arm_speed[tid] + 0.35 * speed
                    )

                    lying, info = is_lying(box, kpts)
                    color = (0, 200, 0)
                    label = f"person #{tid}"

                    if lying:
                        posture_since.setdefault(tid, now)
                        posture_confirmed = (
                            now - posture_since[tid] >= args.posture_confirm_seconds
                        )
                        if posture_confirmed:
                            lying_since.setdefault(tid, now)
                            elapsed = now - lying_since[tid]
                            drop = estimate_drop(
                                hist, now, center[1], size
                            )

                            if drop is not None and drop >= args.fall_drop:
                                if tid not in fall_alerted:
                                    trigger(
                                        "possible_fall",
                                        f"Possible fall pattern for track {tid} "
                                        f"(estimated drop {drop:.2f} body sizes)",
                                        frame
                                    )
                                    fall_alerted.add(tid)

                            if elapsed >= args.down_seconds:
                                color = (0, 0, 255)
                                label = f"REVIEW: PERSON DOWN? #{tid} ({elapsed:.0f}s)"
                                if tid not in person_down_alerted:
                                    person_down_alerted.add(tid)
                                    trigger(
                                        "person_down",
                                        f"Track {tid} remained in a horizontal posture "
                                        f"for {elapsed:.1f}s",
                                        frame
                                    )
                            else:
                                color = (0, 255, 255)
                                label = f"horizontal posture #{tid} ({elapsed:.1f}s)"
                        else:
                            color = (0, 200, 255)
                            label = f"checking posture #{tid}"
                    else:
                        posture_since.pop(tid, None)
                        lying_since.pop(tid, None)
                        person_down_alerted.discard(tid)
                        fall_alerted.discard(tid)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    draw_label(frame, label, x1, y1 - 7, color)

                    if args.debug:
                        draw_label(frame, info, x1, min(y2 + 18, frame.shape[0] - 5),
                                   (255, 255, 255))

            last_people_count = people_count

            # Clear stale tracks and all related state consistently.
            expired = [
                tid for tid, seen_at in list(last_seen.items())
                if now - seen_at > args.track_timeout
            ]
            for tid in expired:
                last_seen.pop(tid, None)
                posture_since.pop(tid, None)
                lying_since.pop(tid, None)
                person_down_alerted.discard(tid)
                fall_alerted.discard(tid)
                prev_rel.pop(tid, None)
                arm_speed.pop(tid, None)
                y_hist.pop(tid, None)

            # Pair proximity + sustained motion is only a review cue, not
            # a reliable way to classify an interaction.
            active_pairs = set()
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    close = (
                        np.linalg.norm(centers[i] - centers[j])
                        < args.interaction_dist * max(sizes[i], sizes[j])
                    )
                    movement_present = (
                        max(arm_speed[ids[i]], arm_speed[ids[j]])
                        > args.interaction_speed
                    )
                    if not (close and movement_present):
                        continue

                    pair = tuple(sorted((ids[i], ids[j])))
                    active_pairs.add(pair)
                    interaction_since.setdefault(pair, now)

                    if now - interaction_since[pair] >= args.interaction_seconds:
                        bx = np.array([boxes_i[i], boxes_i[j]])
                        left = int(bx[:, 0].min())
                        top = int(bx[:, 1].min())
                        right = int(bx[:, 2].max())
                        bottom = int(bx[:, 3].max())
                        cv2.rectangle(frame, (left, top), (right, bottom),
                                      (0, 0, 255), 2)
                        draw_label(
                            frame, "REVIEW: UNUSUAL CLOSE INTERACTION",
                            left, top - 8, (0, 0, 255)
                        )
                        if pair not in interaction_alerted:
                            interaction_alerted.add(pair)
                            trigger(
                                "interaction_review",
                                f"Sustained close-range movement involving tracks "
                                f"{pair[0]} and {pair[1]}",
                                frame
                            )

            for pair in list(interaction_since):
                if pair not in active_pairs:
                    interaction_since.pop(pair, None)
                    interaction_alerted.discard(pair)

            # Group gathering is a neutral caution indicator, not an incident.
            if len(centers) >= args.group_size:
                for group in find_groups(
                    centers, sizes, args.group_size, args.group_dist
                ):
                    pts = np.array([centers[i] for i in group])
                    gx1, gy1 = pts.min(axis=0).astype(int) - 25
                    gx2, gy2 = pts.max(axis=0).astype(int) + 25
                    gx1, gy1 = max(gx1, 0), max(gy1, 0)
                    gx2 = min(gx2, frame.shape[1] - 1)
                    gy2 = min(gy2, frame.shape[0] - 1)
                    cv2.rectangle(frame, (gx1, gy1), (gx2, gy2),
                                  (0, 140, 255), 2)
                    draw_label(
                        frame, f"GROUP: {len(group)} people",
                        gx1, gy1 - 7, (0, 140, 255)
                    )
                    if now - last_group_log >= args.group_log_interval:
                        last_group_log = now
                        print(
                            f"[{datetime.now().strftime('%H:%M:%S')}] "
                            f"Group caution: {len(group)} people"
                        )

            ai_fps = 0.0
            if len(frame_times) > 1 and frame_times[-1] > frame_times[0]:
                ai_fps = (len(frame_times) - 1) / (
                    frame_times[-1] - frame_times[0]
                )

            cv2.putText(
                frame,
                f"People: {last_people_count} | AI FPS: {ai_fps:.1f} "
                f"| Inference: {last_inference_ms:.0f} ms",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (255, 255, 255), 2, cv2.LINE_AA
            )

            if now - last_stats_print >= 5.0:
                last_stats_print = now
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"AI FPS: {ai_fps:.1f} | people: {last_people_count} "
                    f"| inference: {last_inference_ms:.0f} ms"
                )

            if not args.no_show:
                cv2.imshow("Drone Safety Monitor - UNO Q", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lightweight pose-based safety monitor for Arduino UNO Q Linux MPU."
    )
    parser.add_argument("--source", default="0",
                        help="Camera index, video file, or stream URL")
    parser.add_argument("--width", type=int, default=480,
                        help="Processing frame width; use 0 to keep original width")
    parser.add_argument("--model", default="yolov8n-pose.pt",
                        help="Use the lightweight YOLOv8 Nano pose model on UNO Q")
    parser.add_argument("--tracker", default="bytetrack.yaml",
                        help="Ultralytics tracker configuration")
    parser.add_argument("--device", default="cpu",
                        help="UNO Q default is CPU; change only after verifying another backend")
    parser.add_argument("--half", action="store_true",
                        help="Optional FP16; do not use with the default CPU device")
    parser.add_argument("--imgsz", type=int, default=416,
                        help="Smaller input reduces compute and memory; try 320 or 416")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=30)
    parser.add_argument("--skip-frames", type=int, default=1,
                        help="Run inference on 1 of every N+1 frames")
    parser.add_argument("--show-pose", action="store_true",
                        help="Draw pose keypoints and skeletons")
    parser.add_argument("--debug", action="store_true",
                        help="Show posture diagnostics")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--no-beep", action="store_true")
    parser.add_argument("--alert-dir", default="alerts")

    parser.add_argument("--posture-confirm-seconds", type=float, default=0.6,
                        help="How long horizontal posture must persist before timing")
    parser.add_argument("--down-seconds", type=float, default=5.0,
                        help="Seconds in horizontal posture before person-down review alert")
    parser.add_argument("--fall-drop", type=float, default=0.35,
                        help="Recent downward displacement in body sizes")
    parser.add_argument("--track-timeout", type=float, default=3.0,
                        help="Seconds without a track before its state is cleared")

    parser.add_argument("--interaction-seconds", type=float, default=2.0,
                        help="Duration before a close-movement review alert")
    parser.add_argument("--interaction-speed", type=float, default=2.0,
                        help="Relative wrist-motion threshold")
    parser.add_argument("--interaction-dist", type=float, default=1.0,
                        help="Normalized distance threshold for close interaction")

    parser.add_argument("--group-size", type=int, default=5)
    parser.add_argument("--group-dist", type=float, default=1.5)
    parser.add_argument("--group-log-interval", type=float, default=10.0)

    run(parser.parse_args())
