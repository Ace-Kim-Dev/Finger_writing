import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

HAND_MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")
CAMERA_INDEX = 0

# Your preview is mirrored. This fixes Left/Right for your view.
SWAP_HANDEDNESS_FOR_MIRROR = True

# Writing stability
SMOOTH_ALPHA = 0.72
MAX_JUMP_PX = 95
PINCH_START_RATIO = 0.40
PINCH_END_RATIO = 0.52
TRAIL_FADE = 0.992

FINGER_TIPS = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# OpenCV colors use BGR.
TIP_COLORS = {
    "thumb": (255, 120, 40),
    "index": (255, 0, 255),
    "middle": (0, 255, 255),
    "ring": (0, 120, 255),
    "pinky": (0, 255, 80),
}

HAND_COLORS = {
    "Left": (255, 120, 255),
    "Right": (255, 220, 80),
}

CLAHE = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))


def normalize_lighting(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    corrected = CLAHE.apply(lightness)
    return cv2.cvtColor(
        cv2.merge((corrected, a_channel, b_channel)),
        cv2.COLOR_LAB2BGR,
    )


def to_pixel(landmark, width, height):
    x = int(np.clip(round(landmark.x * (width - 1)), 0, width - 1))
    y = int(np.clip(round(landmark.y * (height - 1)), 0, height - 1))
    return np.array((x, y), dtype=np.int32)


def hand_label(result, index):
    try:
        label = result.handedness[index][0].category_name
    except (IndexError, AttributeError):
        label = f"Hand {index + 1}"

    if SWAP_HANDEDNESS_FOR_MIRROR:
        return {"Left": "Right", "Right": "Left"}.get(label, label)

    return label


class LandmarkSmoother:
    def __init__(self, alpha):
        self.alpha = alpha
        self.previous = {}

    def update(self, hand_id, points):
        smoothed = []

        for index, point in enumerate(points):
            key = (hand_id, index)
            old = self.previous.get(key)

            if old is None:
                value = point.astype(np.float32)
            else:
                value = self.alpha * point + (1 - self.alpha) * old

            self.previous[key] = value
            smoothed.append(np.rint(value).astype(np.int32))

        return smoothed

    def forget_missing_hands(self, active_hand_ids):
        for key in list(self.previous):
            if key[0] not in active_hand_ids:
                del self.previous[key]


class PinchPen:
    """A thumb-index pinch means the pen is touching the virtual canvas."""

    def __init__(self):
        self.is_down = {}

    def update(self, hand_id, points):
        thumb_tip = points[4]
        index_tip = points[8]

        # Scale pinch distance to hand size, so it works nearer/farther from camera.
        palm_width = max(float(np.linalg.norm(points[5] - points[17])), 30.0)
        pinch_ratio = float(np.linalg.norm(thumb_tip - index_tip)) / palm_width

        already_down = self.is_down.get(hand_id, False)
        threshold = PINCH_END_RATIO if already_down else PINCH_START_RATIO

        pen_down = pinch_ratio < threshold
        self.is_down[hand_id] = pen_down

        return pen_down

    def forget_missing_hands(self, active_hand_ids):
        for hand_id in list(self.is_down):
            if hand_id not in active_hand_ids:
                del self.is_down[hand_id]


class InkCanvas:
    def __init__(self):
        self.canvas = None
        self.previous = {}

    def ensure_size(self, frame):
        if self.canvas is None or self.canvas.shape != frame.shape:
            self.canvas = np.zeros_like(frame)
            self.previous.clear()

    def clear(self):
        self.canvas[:] = 0
        self.previous.clear()

    def stop_strokes(self):
        self.previous.clear()

    def fade(self, fade_on):
        if fade_on:
            self.canvas = (
                self.canvas.astype(np.float32) * TRAIL_FADE
            ).astype(np.uint8)

    def draw_writing(self, hands, drawing_on):
        """Only index draws, and only while thumb + index are pinched."""
        active_strokes = set()

        if drawing_on:
            for hand in hands:
                if not hand["pen_down"]:
                    continue

                key = f"{hand['id']}:index"
                active_strokes.add(key)

                point = hand["tips"]["index"]
                previous = self.previous.get(key)

                if previous is not None and np.linalg.norm(point - previous) < MAX_JUMP_PX:
                    cv2.line(
                        self.canvas,
                        tuple(previous),
                        tuple(point),
                        TIP_COLORS["index"],
                        6,
                        cv2.LINE_AA,
                    )

                cv2.circle(
                    self.canvas,
                    tuple(point),
                    4,
                    TIP_COLORS["index"],
                    -1,
                    cv2.LINE_AA,
                )

                self.previous[key] = point.copy()

        # Releasing the pinch ends the stroke, so it cannot draw a line on return.
        for key in list(self.previous):
            if key not in active_strokes:
                del self.previous[key]

    def draw_art(self, hands, shapes_on, drawing_on):
        """Every fingertip paints; optional shapes merge the fingertips."""
        active_strokes = set()

        if drawing_on:
            for hand in hands:
                for finger, point in hand["tips"].items():
                    key = f"{hand['id']}:{finger}"
                    active_strokes.add(key)

                    previous = self.previous.get(key)

                    if previous is not None and np.linalg.norm(point - previous) < MAX_JUMP_PX:
                        cv2.line(
                            self.canvas,
                            tuple(previous),
                            tuple(point),
                            TIP_COLORS[finger],
                            5,
                            cv2.LINE_AA,
                        )

                    cv2.circle(
                        self.canvas,
                        tuple(point),
                        3,
                        TIP_COLORS[finger],
                        -1,
                        cv2.LINE_AA,
                    )

                    self.previous[key] = point.copy()

                # This only runs in ART mode, never writing mode.
                if shapes_on:
                    polygon = np.array(
                        list(hand["tips"].values()),
                        dtype=np.int32,
                    )
                    hull = cv2.convexHull(polygon)

                    if len(hull) >= 3 and cv2.contourArea(hull) > 250:
                        layer = np.zeros_like(self.canvas)
                        color = HAND_COLORS.get(hand["id"], (200, 200, 255))

                        cv2.fillConvexPoly(layer, hull, color)
                        self.canvas = cv2.addWeighted(
                            self.canvas,
                            1.0,
                            layer,
                            0.11,
                            0,
                        )
                        cv2.polylines(
                            self.canvas,
                            [hull],
                            True,
                            color,
                            2,
                            cv2.LINE_AA,
                        )

        for key in list(self.previous):
            if key not in active_strokes:
                del self.previous[key]

    def render(self, frame):
        glow = cv2.GaussianBlur(self.canvas, (0, 0), 11)
        output = cv2.addWeighted(frame, 1.0, glow, 0.35, 0)
        return cv2.add(output, self.canvas)


def draw_hand(frame, hand, show_labels):
    points = hand["points"]
    label = hand["id"]
    color = HAND_COLORS.get(label, (255, 255, 255))

    for first, second in HAND_CONNECTIONS:
        cv2.line(
            frame,
            tuple(points[first]),
            tuple(points[second]),
            color,
            2,
            cv2.LINE_AA,
        )

    for name, tip_index in FINGER_TIPS.items():
        point = points[tip_index]

        cv2.circle(
            frame,
            tuple(point),
            7,
            TIP_COLORS[name],
            -1,
            cv2.LINE_AA,
        )

        if show_labels:
            cv2.putText(
                frame,
                name,
                (int(point[0]) + 8, int(point[1]) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                TIP_COLORS[name],
                1,
                cv2.LINE_AA,
            )

    status = "PEN DOWN" if hand["pen_down"] else "aiming"
    status_color = (0, 255, 0) if hand["pen_down"] else color

    cv2.putText(
        frame,
        f"{label}: {status}",
        (int(points[0][0]) + 8, int(points[0][1]) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        status_color,
        2,
        cv2.LINE_AA,
    )


def draw_hud(frame, fps, mode, shapes_on, fade_on, drawing_on, lighting_on):
    mode_line = (
        "WRITE: pinch thumb + index to draw"
        if mode == "write"
        else "ART: every fingertip paints"
    )

    lines = [
        f"FPS: {fps:.1f}",
        mode_line,
        f"shapes: {'ON' if shapes_on and mode == 'art' else 'OFF'}",
        f"drawing: {'ON' if drawing_on else 'PAUSED'} | "
        f"fade: {'ON' if fade_on else 'OFF'} | "
        f"lighting: {'ON' if lighting_on else 'OFF'}",
    ]

    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (380, 18 + 22 * len(lines)), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.60, frame, 0.40, 0)

    for row, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (16, 31 + row * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    controls = (
        "q quit | c clear | 1 write | 2 art | s shapes | f fade | "
        "space pause | l lighting | v labels"
    )

    cv2.putText(
        frame,
        controls,
        (10, frame.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def open_camera():
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    camera = cv2.VideoCapture(CAMERA_INDEX, backend)

    if not camera.isOpened() and backend != cv2.CAP_ANY:
        camera = cv2.VideoCapture(CAMERA_INDEX)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_FPS, 30)

    return camera


def main():
    if not HAND_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {HAND_MODEL_PATH.name}. Put it beside this script."
        )

    camera = open_camera()

    if not camera.isOpened():
        raise RuntimeError(f"Could not open webcam {CAMERA_INDEX}.")

    BaseOptions = mp.tasks.BaseOptions
    vision = mp.tasks.vision

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,

        # Less likely to drop your hand during fast motion.
        min_hand_detection_confidence=0.50,
        min_hand_presence_confidence=0.45,
        min_tracking_confidence=0.50,
    )

    smoother = LandmarkSmoother(SMOOTH_ALPHA)
    pinch_pen = PinchPen()
    ink = InkCanvas()

    mode = "write"       # Starts in air-writing mode.
    shapes_on = True     # Only applies to art mode.
    fade_on = False      # Writing stays until you clear it.
    drawing_on = True
    lighting_on = False  # Natural camera image is usually better for hand tracking.
    show_labels = True

    start_time = time.monotonic()
    last_timestamp = -1
    previous_frame_time = time.monotonic()
    fps = 0.0

    try:
        with vision.HandLandmarker.create_from_options(options) as tracker:
            while camera.isOpened():
                ok, frame = camera.read()

                if not ok:
                    print("Camera feed lost")
                    break

                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]

                ink.ensure_size(frame)

                model_input = normalize_lighting(frame) if lighting_on else frame
                rgb = np.ascontiguousarray(
                    cv2.cvtColor(model_input, cv2.COLOR_BGR2RGB)
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )

                timestamp_ms = int((time.monotonic() - start_time) * 1000)
                timestamp_ms = max(timestamp_ms, last_timestamp + 1)
                last_timestamp = timestamp_ms

                result = tracker.detect_for_video(mp_image, timestamp_ms)

                hands = []
                active_hand_ids = set()

                for index, landmarks in enumerate(result.hand_landmarks):
                    label = hand_label(result, index)

                    raw_points = [
                        to_pixel(landmark, width, height)
                        for landmark in landmarks
                    ]

                    points = smoother.update(label, raw_points)
                    pen_down = pinch_pen.update(label, points)

                    tips = {
                        name: points[tip_index]
                        for name, tip_index in FINGER_TIPS.items()
                    }

                    hands.append({
                        "id": label,
                        "points": points,
                        "tips": tips,
                        "pen_down": pen_down,
                    })

                    active_hand_ids.add(label)

                smoother.forget_missing_hands(active_hand_ids)
                pinch_pen.forget_missing_hands(active_hand_ids)

                ink.fade(fade_on)

                if mode == "write":
                    ink.draw_writing(hands, drawing_on)
                else:
                    ink.draw_art(hands, shapes_on, drawing_on)

                output = ink.render(frame)

                for hand in hands:
                    draw_hand(output, hand, show_labels)

                now = time.monotonic()
                instant_fps = 1 / max(now - previous_frame_time, 1e-6)
                fps = (
                    instant_fps
                    if fps == 0
                    else 0.15 * instant_fps + 0.85 * fps
                )
                previous_frame_time = now

                draw_hud(
                    output,
                    fps,
                    mode,
                    shapes_on,
                    fade_on,
                    drawing_on,
                    lighting_on,
                )

                cv2.imshow("Air Writer + Finger Trails", output)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                elif key == ord("c"):
                    ink.clear()
                elif key == ord("1"):
                    mode = "write"
                    ink.stop_strokes()
                elif key == ord("2"):
                    mode = "art"
                    ink.stop_strokes()
                elif key == ord("s"):
                    shapes_on = not shapes_on
                elif key == ord("f"):
                    fade_on = not fade_on
                elif key == ord("l"):
                    lighting_on = not lighting_on
                elif key == ord("v"):
                    show_labels = not show_labels
                elif key == 32:
                    drawing_on = not drawing_on
                    ink.stop_strokes()

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()