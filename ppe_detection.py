import cv2
import re
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from ultralytics import YOLO

# ─── Constants ────────────────────────────────────────────────────────────────
CONF_THRESHOLD = 0.4       # Ignore detections below this confidence
DEFAULT_IMGSZ = 640        # YOLO inference size (smaller = faster, less accurate on small objects)
FONT_SCALE = 0.6
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FALLBACK_WRITE_FPS = 30.0

# Consecutive violation frames before triggering an alert log
ALERT_FRAME_THRESHOLD = 10

# BGR colors
COLOR_VIOLATION = (0, 0, 255)    # Red
COLOR_COMPLIANT = (0, 255, 0)    # Green
COLOR_PERSON = (255, 0, 0)       # Blue
COLOR_OTHER = (0, 255, 255)      # Yellow
COLOR_HUD_BG = (0, 0, 0)
COLOR_FPS = (0, 255, 255)
COLOR_ALERT_OK = (0, 200, 0)

# Explicit class-name mappings for the trained best.pt labels:
# {0: Hardhat, 1: Mask, 2: NO-Hardhat, 3: NO-Mask, 4: NO-Safety Vest,
#  5: Person, 6: Safety Cone, 7: Safety Vest, 8: machinery, 9: vehicle}
#
# Matching uses a normalized form (lowercase, hyphens/underscores -> spaces).
# These are exact normalized names, not broad substring checks.
VIOLATION_CLASS_NAMES = {
    "no hardhat",
    "no helmet",
    "no mask",
    "no safety vest",
    "no vest",
    "without hardhat",
    "without helmet",
    "without mask",
    "without safety vest",
    "without vest",
}
COMPLIANT_PPE_CLASS_NAMES = {
    "hardhat",
    "helmet",
    "mask",
    "safety vest",
    "vest",
    "glove",
    "gloves",
}
PERSON_CLASS_NAMES = {
    "person",
    "worker",
}
# Tokens that indicate a PPE item when combined with a leading "no" / "without".
PPE_ITEM_TOKENS = {
    "hardhat", "helmet", "mask", "vest", "glove", "gloves", "ppe",
}

Category = str  # "violation" | "compliant" | "person" | "other"
# ──────────────────────────────────────────────────────────────────────────────


def normalize_class_name(cls_name: str) -> str:
    """Lowercase and collapse hyphens/underscores/spaces so names compare reliably."""
    name = str(cls_name).strip().lower().replace("-", " ").replace("_", " ")
    return re.sub(r"\s+", " ", name)


def classify_detection(cls_name: str) -> Category:
    """
    Map a model class name to a detection category.

    - violation: explicit missing-PPE classes (e.g. NO-Hardhat)
    - compliant: PPE-present classes (e.g. Hardhat, Safety Vest)
    - person: Person / Worker (not a PPE compliance result)
    - other: cones, vehicles, machinery, unknown labels

    Limitation: a compliant PPE box means the model detected that PPE class; it
    does not prove a specific worker is wearing it, because this script does
    not associate PPE boxes with Person boxes.
    """
    name = normalize_class_name(cls_name)
    if not name:
        return "other"

    if name in VIOLATION_CLASS_NAMES:
        return "violation"
    if name in COMPLIANT_PPE_CLASS_NAMES:
        return "compliant"
    if name in PERSON_CLASS_NAMES:
        return "person"

    tokens = name.split()
    # Word-boundary "no"/"without" as a leading token only (avoids matching "normal").
    if len(tokens) >= 2 and tokens[0] in {"no", "without"}:
        remainder_tokens = set(tokens[1:])
        if remainder_tokens & PPE_ITEM_TOKENS:
            return "violation"

    return "other"


def get_ppe_color(cls_name: str) -> Tuple[int, int, int]:
    """
    Returns a BGR color based on the class name:
      RED    -> PPE violation (NO-Hardhat, NO-Mask, NO-Safety Vest, …)
      GREEN  -> PPE item class the model treats as present (Hardhat, Mask, …)
      BLUE   -> Person / Worker
      YELLOW -> Everything else (Safety Cone, machinery, vehicle, unknown)
    """
    category = classify_detection(cls_name)
    if category == "violation":
        return COLOR_VIOLATION
    if category == "compliant":
        return COLOR_COMPLIANT
    if category == "person":
        return COLOR_PERSON
    return COLOR_OTHER


def is_violation(cls_name: str) -> bool:
    """True when the class name is a PPE-violation detection."""
    return classify_detection(cls_name) == "violation"


def get_class_name(model: Any, cls_id: int) -> str:
    if isinstance(model.names, dict):
        return str(model.names.get(cls_id, cls_id))
    return str(model.names[cls_id])


def get_model_class_names(model: Any) -> List[str]:
    names = model.names
    if isinstance(names, dict):
        keys = sorted(names.keys(), key=lambda k: int(k) if str(k).isdigit() else str(k))
        return [str(names[k]) for k in keys]
    return [str(n) for n in names]


def print_class_mappings(class_names: Iterable[str]) -> None:
    print("[INFO] Class mappings (verified against model.names):")
    for raw in class_names:
        category = classify_detection(raw)
        print(f"         {raw!r:20} -> {category}")
    print(
        "[INFO] Counts in the HUD are detection boxes, not unique workers. "
        "PPE items are not associated with Person boxes."
    )


def resolve_device(device: str) -> str:
    """Pick CUDA:0 when available unless the user requested a specific device."""
    requested = (device or "auto").strip().lower()
    if requested not in {"auto", ""}:
        return device
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        print("[WARN] torch is not importable; falling back to CPU.")
        return "cpu"


def _text_size(text: str, font_scale: float, thickness: int) -> Tuple[int, int]:
    (tw, th), _ = cv2.getTextSize(text, FONT, font_scale, thickness)
    return tw, th


def draw_hud(frame, fps: float, violations: int, safe: int, alert: bool) -> None:
    """Draw FPS, detection counts, and alert status on a semi-transparent HUD."""
    h, w = frame.shape[:2]
    scale = max(0.55, min(1.15, w / 1280.0))
    font_scale = 0.55 * scale
    thickness = max(1, int(round(2 * scale)))
    pad_x = int(10 * scale)
    pad_y = int(8 * scale)
    line_gap = int(8 * scale)

    alert_text = "ALERT ACTIVE" if alert else "NO ACTIVE ALERT"
    lines: List[Tuple[str, Tuple[int, int, int]]] = [
        (f"FPS              : {fps:.1f}", COLOR_FPS),
        (f"PPE violations   : {violations} dets", COLOR_VIOLATION),
        (f"Safe PPE         : {safe} dets", COLOR_COMPLIANT),
        (f"Alert            : {alert_text}", COLOR_VIOLATION if alert else COLOR_ALERT_OK),
    ]

    sizes = [_text_size(text, font_scale, thickness) for text, _ in lines]
    text_w = max(sz[0] for sz in sizes)
    text_h = max(sz[1] for sz in sizes)
    line_h = text_h + line_gap
    hud_w = min(w, text_w + pad_x * 2)
    hud_h = min(h, pad_y * 2 + line_h * len(lines) - line_gap)

    # Blend only the HUD region to avoid copying the full frame every loop.
    roi = frame[0:hud_h, 0:hud_w]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (hud_w, hud_h), COLOR_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.45, roi, 0.55, 0, roi)

    y = pad_y + text_h
    for (text, color), _ in zip(lines, sizes):
        cv2.putText(frame, text, (pad_x, y), FONT, font_scale, color, thickness, cv2.LINE_AA)
        y += line_h


def draw_alert_banner(frame, active: bool) -> None:
    """Full-width warning bar at the bottom when a consecutive-frame alert is active."""
    if not active:
        return
    h, w = frame.shape[:2]
    scale = max(0.55, min(1.2, w / 1280.0))
    font_scale = 0.8 * scale
    thickness = max(2, int(round(2 * scale)))
    bar_h = max(36, int(48 * scale))
    y0 = max(0, h - bar_h)

    roi = frame[y0:h, 0:w]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 180), -1)
    cv2.addWeighted(overlay, 0.55, roi, 0.45, 0, roi)

    text = "ALERT: PPE VIOLATION (detection count, not a unique worker)"
    tw, th = _text_size(text, font_scale, thickness)
    x = max(10, (w - tw) // 2)
    y = y0 + (bar_h + th) // 2
    cv2.putText(frame, text, (x, y), FONT, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


@dataclass
class AlertState:
    streak: int = 0
    active: bool = False
    logged_this_streak: bool = False
    last_notify_time: float = 0.0


def update_alert_state(
    state: AlertState,
    has_violation: bool,
    threshold: int,
    now: float,
    cooldown_s: float = 0.0,
) -> bool:
    """
    Consecutive-frame alert.

    - Activates when a violation is present for `threshold` frames in a row.
    - Resets immediately when a frame has no violation detections.
    - Returns True when a notification should fire (first activation, or
      optional cooldown re-notify during a continuing streak).
    """
    should_notify = False
    if has_violation:
        state.streak += 1
        if state.streak >= threshold:
            if not state.active:
                state.active = True
                should_notify = True
            elif cooldown_s > 0 and (now - state.last_notify_time) >= cooldown_s:
                should_notify = True
        if should_notify:
            state.last_notify_time = now
            state.logged_this_streak = True
    else:
        state.streak = 0
        state.active = False
        state.logged_this_streak = False
    return should_notify


_sound_unavailable_logged = False


def play_alert_sound(enabled: bool) -> None:
    """Optional short beep. Detection continues if audio support is missing."""
    global _sound_unavailable_logged
    if not enabled:
        return
    try:
        import winsound
        winsound.Beep(1200, 180)
    except Exception as exc:  # ImportError, RuntimeError, etc.
        if not _sound_unavailable_logged:
            print(f"[WARN] Sound notification unavailable ({exc}). Continuing without audio.")
            _sound_unavailable_logged = True


def setup_video_writer(
    cap: cv2.VideoCapture,
    output_path: str,
    fallback_fps: float = FALLBACK_WRITE_FPS,
) -> Tuple[Optional[cv2.VideoWriter], Tuple[int, int], float]:
    """
    Create a VideoWriter matching the source resolution.

    Returns (writer_or_None, (width, height), fps).
    """
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    if width <= 0 or height <= 0:
        print(f"[ERROR] Invalid capture size: {width}x{height}")
        return None, (width, height), fallback_fps

    fps = raw_fps
    if not (1.0 <= fps <= 120.0):
        print(
            f"[WARN] Source FPS is {raw_fps:.2f}; using fallback {fallback_fps:.1f} FPS for recording."
        )
        fps = fallback_fps

    out = Path(output_path)
    try:
        if out.parent and str(out.parent) not in {".", ""}:
            out.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"[ERROR] Cannot create output directory for {output_path}: {exc}")
        return None, (width, height), fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"[ERROR] Failed to open video writer for: {output_path}")
        return None, (width, height), fps

    print(f"[INFO] Video writer ready: {out} ({width}x{height} @ {fps:.1f} FPS)")
    return writer, (width, height), fps


def write_frame(
    writer: cv2.VideoWriter,
    frame,
    size: Tuple[int, int],
) -> None:
    """Write a frame, resizing only when it does not match the writer size."""
    height, width = frame.shape[:2]
    target_w, target_h = size
    if width != target_w or height != target_h:
        frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    writer.write(frame)


def parse_source(source: str) -> Union[int, str]:
    return int(source) if source.isdigit() else source


def run_detection(
    model_path: str,
    source: str,
    save: bool,
    output_path: str,
    conf: float = CONF_THRESHOLD,
    imgsz: int = DEFAULT_IMGSZ,
    device: str = "auto",
    alert_frames: int = ALERT_FRAME_THRESHOLD,
    sound: bool = False,
    cooldown_s: float = 0.0,
) -> None:
    # ── Load Model ──────────────────────────────────────────────────────────
    print(f"\n[INFO] Loading model : {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    class_names = get_model_class_names(model)
    print(f"[INFO] Model classes : {class_names}")
    print_class_mappings(class_names)

    resolved_device = resolve_device(device)
    print(f"[INFO] Inference device : {resolved_device}")
    print(
        "[INFO] Speed/accuracy: lower --imgsz and higher --conf are faster / fewer boxes; "
        f"current imgsz={imgsz}, conf={conf:.2f}."
    )

    # ── Open Source ──────────────────────────────────────────────────────────
    src = parse_source(source)
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        return

    print(f"[INFO] Source opened : {source}")
    print("[INFO] Press Q to quit\n")

    # ── Video Writer (optional) ──────────────────────────────────────────────
    writer: Optional[cv2.VideoWriter] = None
    writer_size = (0, 0)
    if save:
        writer, writer_size, _ = setup_video_writer(cap, output_path)
        if writer is None:
            print("[WARN] Recording disabled because the video writer did not open.")
        else:
            print(f"[INFO] Saving output to : {output_path}")

    # ── State ────────────────────────────────────────────────────────────────
    prev_time = time.perf_counter()
    fps_display = 0.0
    alert_state = AlertState()
    predict_kwargs: Dict[str, Any] = {
        "verbose": False,
        "conf": conf,
        "imgsz": imgsz,
        "device": resolved_device,
    }

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[INFO] End of stream or failed to grab frame.")
                break

            # ── Inference ────────────────────────────────────────────────────
            try:
                results = model(frame, **predict_kwargs)
            except Exception as e:
                print(f"[ERROR] Inference failed: {e}")
                break

            # ── Process Detections ───────────────────────────────────────────
            # These are bounding-box counts, not unique worker counts.
            violation_count = 0
            safe_count = 0

            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    cls_name = get_class_name(model, cls_id)

                    # Show only the six PPE classes relevant to this project.
                    allowed_ppe_classes = {
                        "hardhat",
                        "no hardhat",
                        "mask",
                        "no mask",
                        "safety vest",
                        "no safety vest",
                    }
                    if normalize_class_name(cls_name) not in allowed_ppe_classes:
                        continue

                    category = classify_detection(cls_name)
                    color = get_ppe_color(cls_name)
                    conf_val = float(box.conf[0])
                    label = f"{cls_name} {conf_val:.2f}"

                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
                    cv2.putText(
                        frame, label,
                        (x1, max(y1 - 6, 10)),
                        FONT, FONT_SCALE, color, BOX_THICKNESS,
                    )

                    if category == "violation":
                        violation_count += 1
                    elif category == "compliant":
                        safe_count += 1

            # ── Alert Logic ──────────────────────────────────────────────────
            now = time.time()
            should_notify = update_alert_state(
                alert_state,
                has_violation=violation_count > 0,
                threshold=alert_frames,
                now=now,
                cooldown_s=cooldown_s,
            )
            if should_notify:
                print(
                    f"[ALERT] {time.strftime('%Y-%m-%d %H:%M:%S')} — PPE violation "
                    f"detected for {alert_frames} consecutive frames "
                    f"({violation_count} violation detection(s) this frame)."
                )
                play_alert_sound(sound)

            # ── FPS Calculation (EMA of instantaneous 1/dt) ──────────────────
            curr_time = time.perf_counter()
            instant_fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time
            fps_display = instant_fps if fps_display == 0.0 else (0.85 * fps_display + 0.15 * instant_fps)

            # ── Draw HUD ─────────────────────────────────────────────────────
            draw_hud(frame, fps_display, violation_count, safe_count, alert_state.active)
            draw_alert_banner(frame, alert_state.active)

            # ── Display & Save ───────────────────────────────────────────────
            cv2.imshow("PPE Detection (press Q to quit)", frame)
            if writer is not None:
                try:
                    write_frame(writer, frame, writer_size)
                except Exception as e:
                    print(f"[ERROR] Failed to write output frame: {e}")
                    writer.release()
                    writer = None

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Quit key pressed.")
                break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"[INFO] Output saved to: {output_path}")
        cv2.destroyAllWindows()
        print("[INFO] Detection ended.")


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv9 PPE Real-time Detection")

    parser.add_argument(
        "--model", type=str, default="best.pt",
        help="Path to YOLO model weights (default: best.pt)",
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Camera index (0, 1 ...) or path to video file (default: 0)",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save the output video",
    )
    parser.add_argument(
        "--output", type=str, default="output.mp4",
        help="Output video filename when --save is used (default: output.mp4)",
    )
    parser.add_argument(
        "--conf", type=float, default=CONF_THRESHOLD,
        help=f"Confidence threshold (default: {CONF_THRESHOLD}). Higher = fewer boxes, faster NMS.",
    )
    parser.add_argument(
        "--imgsz", type=int, default=DEFAULT_IMGSZ,
        help=f"Inference image size (default: {DEFAULT_IMGSZ}). Smaller is faster; larger can help small PPE.",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Inference device: auto, cpu, 0 (CUDA:0), 1, … (default: auto)",
    )
    parser.add_argument(
        "--alert-frames", type=int, default=ALERT_FRAME_THRESHOLD,
        help=f"Consecutive violation frames before an alert (default: {ALERT_FRAME_THRESHOLD})",
    )
    parser.add_argument(
        "--sound", action="store_true",
        help="Play a short beep when an alert is first triggered (optional)",
    )
    parser.add_argument(
        "--alert-cooldown", type=float, default=0.0,
        help="Seconds before re-notifying during one continuous streak (default: 0 = once per streak)",
    )

    args = parser.parse_args()

    run_detection(
        model_path=args.model,
        source=args.source,
        save=args.save,
        output_path=args.output,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        alert_frames=args.alert_frames,
        sound=args.sound,
        cooldown_s=args.alert_cooldown,
    )
