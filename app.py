"""
PiCrawler Cockpit - Flask robot control server.
Run: python3 app.py
"""
import os
import json
import time
import threading
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify, Response

# -------------------- Config --------------------
HERE         = os.path.dirname(os.path.abspath(__file__))
DB_FILE      = os.path.join(HERE, "users.json")
CASCADE_FILE = os.path.join(HERE, "haarcascade_frontalface_default.xml")

FACE_FEATURE_LEN = 864
MATCH_THRESHOLD  = 0.85
FACE_TIMEOUT     = 1.5
VIDEO_QUALITY    = 65

HOST = "0.0.0.0"
PORT = 5000
API_TOKEN = os.environ.get("COCKPIT_TOKEN", "").strip()

# -------------------- Hardware --------------------
crawler          = None
sonar            = None
bat_adc          = None
picam2           = None
camera_available = False


def _init_hardware():
    global crawler, sonar, bat_adc, picam2, camera_available

    try:
        from picrawler import Picrawler
        crawler = Picrawler()
        try:
            crawler.do_step("stand", 50)
        except Exception as exc:
            print(f"[MOVE] initial stand failed: {exc}", flush=True)
        print("[MOVE] Picrawler ready", flush=True)
    except Exception as exc:
        print(f"[MOVE] Picrawler unavailable: {exc}", flush=True)
        crawler = None

    try:
        from robot_hat import Ultrasonic, Pin
        sonar = Ultrasonic(Pin("D2"), Pin("D3"))
        print("[SONAR] ready", flush=True)
    except Exception as exc:
        print(f"[SONAR] unavailable: {exc}", flush=True)
        sonar = None

    try:
        from robot_hat import ADC
        bat_adc = ADC("A4")
        print("[ADC] ready", flush=True)
    except Exception as exc:
        print(f"[ADC] unavailable: {exc}", flush=True)
        bat_adc = None

    try:
        from picamera2 import Picamera2
        picam2 = Picamera2()
        camera_available = True
        print("[CAM] Picamera2 ready", flush=True)
    except Exception as exc:
        print(f"[CAM] unavailable: {exc}", flush=True)
        picam2 = None
        camera_available = False


_init_hardware()


# -------------------- Flask --------------------
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

_cam_lock    = threading.RLock()
_users_lock  = threading.RLock()
_status_lock = threading.Lock()
_i2c_busy    = threading.Event()


# -------------------- Motion control --------------------
CONTINUOUS_ACTIONS = {
    "forward":  ("action", "forward"),
    "backward": ("action", "backward"),
    "left":     ("action", "turn left"),
    "right":    ("action", "turn right"),
}
ONE_SHOT_ACTIONS = {
    "stand": ("step", "stand"),
    "sit":   ("step", "sit"),
    "stop":  ("step", "stand"),
}

_active_motion      = None
_active_motion_lock = threading.Lock()


def _run_step(action):
    if crawler is None:
        return
    kind, command = CONTINUOUS_ACTIONS.get(action) or ONE_SHOT_ACTIONS.get(action)
    if kind == "action":
        crawler.do_action(command, 1, 70)
    else:
        crawler.do_step(command, 50 if command == "stand" else 40)


def _motion_worker():
    """Background thread: keeps walking while _active_motion is set."""
    global _active_motion
    last_action = None
    while True:
        with _active_motion_lock:
            action = _active_motion

        if action is None:
            last_action = None
            time.sleep(0.1)
            continue

        _i2c_busy.set()
        try:
            if action != last_action and crawler:
                crawler.do_step("stand", 50)
                time.sleep(0.2)
            _run_step(action)
            last_action = action
        except Exception as exc:
            print(f"[MOVE] {action} error: {exc}", flush=True)
            with _active_motion_lock:
                _active_motion = None
            last_action = None
        finally:
            _i2c_busy.clear()


threading.Thread(target=_motion_worker, daemon=True).start()


# -------------------- Greet session (state machine) --------------------
_greet = {
    "state": "IDLE",           # IDLE | LOOKING | KNOWN_FACE | ASKING | SAVED
    "state_since": 0.0,
    "message": "",
    "message_id": 0,
    "unknown_started": 0.0,
}
_greet_lock = threading.Lock()


def _greet_set(state, message=""):
    with _greet_lock:
        _greet["state"] = state
        _greet["state_since"] = time.time()
        _greet["message"] = message
        _greet["message_id"] += 1
    print(f"[GREET] -> {state}: {message[:80]}", flush=True)


def _look_up():
    """Tilt body UP: front hips lowered, rear hips raised (nose up).
    Verified servo layout on this robot:
      idx 0 = front right hip
      idx 3 = front left  hip
      idx 6 = rear  left  hip
      idx 9 = rear  right hip
    Positive angle = leg moves UP, negative = DOWN.
    So front DOWN + rear UP -> nose tilts UP."""
    if crawler is None:
        return
    _i2c_busy.set()
    try:
        print("[LOOK_UP] stand", flush=True)
        crawler.do_step("stand", 70)
        time.sleep(0.5)

        try:
            base = list(crawler.servo_positions)
        except Exception as e:
            print(f"[LOOK_UP] cannot read positions: {e}", flush=True)
            return

        if len(base) != 12:
            print(f"[LOOK_UP] unexpected servo count: {len(base)}", flush=True)
            return

        # Nose UP = front hips lower, rear hips raise
        target = list(base)
        target[0] = base[0] - 20   # front right hip down
        target[3] = base[3] - 20   # front left  hip down
        target[6] = base[6] + 20   # rear  left  hip up
        target[9] = base[9] + 20   # rear  right hip up
        target = [round(x, 1) for x in target]
        print(f"[LOOK_UP] target: {target}", flush=True)

        crawler.servo_write_all(target)
        time.sleep(0.4)
        print("[LOOK_UP] look-up pose applied", flush=True)

    except Exception as exc:
        print(f"[LOOK_UP] failed: {exc}", flush=True)
    finally:
        _i2c_busy.clear()


def _greet_tick(face_detected, status, name, hobby):
    """Called per frame. Advances the state machine."""
    now = time.time()
    with _greet_lock:
        s  = _greet["state"]
        ss = _greet["state_since"]

        if not face_detected:
            if s == "KNOWN_FACE" and now - ss > 6.0:
                _greet["state"] = "LOOKING"
                _greet["state_since"] = now
                _greet["message"] = "Looking for a face..."
                _greet["message_id"] += 1
            elif s == "SAVED" and now - ss > 6.0:
                _greet["state"] = "LOOKING"
                _greet["state_since"] = now
                _greet["message"] = "Looking for a face..."
                _greet["message_id"] += 1
            elif s == "ASKING" and now - ss > 30.0:
                _greet["state"] = "LOOKING"
                _greet["state_since"] = now
                _greet["message"] = "Looking for a face..."
                _greet["message_id"] += 1
                _greet["unknown_started"] = 0.0
            return

        if s == "LOOKING":
            if status == "KNOWN" and name:
                _greet["state"] = "KNOWN_FACE"
                _greet["state_since"] = now
                _greet["message"] = (
                    f"Hello, {name}! I know you. "
                    f"Your hobby is {hobby or 'Robotics'}."
                )
                _greet["message_id"] += 1
                _greet["unknown_started"] = 0.0
            elif status == "UNKNOWN":
                if _greet["unknown_started"] == 0.0:
                    _greet["unknown_started"] = now
                elif now - _greet["unknown_started"] > 2.0:
                    _greet["state"] = "ASKING"
                    _greet["state_since"] = now
                    _greet["message"] = (
                        "Hello! I have not seen you before. "
                        "What is your name and what do you like to do?"
                    )
                    _greet["message_id"] += 1
                    _greet["unknown_started"] = 0.0


# -------------------- Face DB --------------------
users       = []
user_matrix = None


def _rebuild_matrix():
    global user_matrix
    rows = []
    for user in users:
        hist = np.asarray(user.get("hist", []), dtype=np.float32)
        if hist.size == FACE_FEATURE_LEN and np.all(np.isfinite(hist)):
            rows.append(hist)
    user_matrix = np.vstack(rows) if rows else None


def load_users():
    global users
    if not os.path.isfile(DB_FILE):
        users = []
        _rebuild_matrix()
        return
    try:
        with open(DB_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        users = [
            item for item in data
            if isinstance(item, dict)
            and isinstance(item.get("name"), str)
            and len(item.get("hist", [])) == FACE_FEATURE_LEN
        ] if isinstance(data, list) else []
        _rebuild_matrix()
        print(f"[DB] Loaded {len(users)} profile(s)", flush=True)
    except Exception as exc:
        print(f"[DB] Load error: {exc}", flush=True)
        users = []
        _rebuild_matrix()


def _save_users_locked():
    tmp_path = DB_FILE + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, DB_FILE)
        _rebuild_matrix()
        return True
    except Exception as exc:
        print(f"[DB] Save error: {exc}", flush=True)
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        return False


load_users()


# -------------------- Face cascade --------------------
local_cascade = CASCADE_FILE
cascade_path = local_cascade if os.path.isfile(local_cascade) else (
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    print(f"[AI] WARNING: cascade not loaded: {cascade_path}", flush=True)
else:
    print(f"[AI] Cascade loaded: {cascade_path}", flush=True)


# -------------------- Camera / state --------------------
current_mode        = "OFF"
last_seen_status    = "NONE"
last_user_name      = ""
last_user_hobby     = ""
current_face_hist   = None
last_face_seen_time = 0.0


def _configure_camera(size):
    try:
        config = picam2.create_video_configuration(
            main={"size": size, "format": "BGR888"}
        )
        picam2.configure(config)
        return "BGR"
    except Exception:
        config = picam2.create_video_configuration(
            main={"size": size, "format": "RGB888"}
        )
        picam2.configure(config)
        return "RGB"


def set_camera_state(target_mode):
    global current_mode, current_face_hist

    if target_mode not in ("OFF", "WALK", "GREET"):
        raise ValueError("Invalid camera mode")
    if target_mode != "OFF" and not camera_available:
        raise RuntimeError("Camera is not available")

    with _cam_lock:
        if current_mode == target_mode:
            return
        if current_mode != "OFF" and picam2 is not None:
            try:
                picam2.stop()
            except Exception as exc:
                print(f"[CAM] Stop warning: {exc}", flush=True)
        current_mode = "OFF"
        current_face_hist = None

        if target_mode == "OFF":
            print("[CAM] Camera stopped (OFF)", flush=True)
            return

        size = (320, 240) if target_mode == "WALK" else (640, 480)
        try:
            pixfmt = _configure_camera(size)
            app.config["PIXFMT"] = pixfmt
            picam2.start()
            current_mode = target_mode
            print(f"[CAM] Mode: {target_mode} ({size[0]}x{size[1]}, {pixfmt})", flush=True)
        except Exception:
            current_mode = "OFF"
            try:
                picam2.stop()
            except Exception:
                pass
            raise


# -------------------- Face features --------------------
def extract_face_features(face_bgr):
    face = cv2.resize(face_bgr, (100, 100), interpolation=cv2.INTER_AREA)
    hsv  = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    features = []
    for i in range(3):
        for j in range(3):
            cell = hsv[i*h//3:(i+1)*h//3, j*w//3:(j+1)*w//3]
            hist = cv2.calcHist([cell], [0, 1], None, [12, 8], [0, 180, 0, 256])
            features.extend(hist.flatten())
    features = np.asarray(features, dtype=np.float32)
    return features / (np.linalg.norm(features) + 1e-7)


def match_face(face_bgr):
    global last_seen_status, last_user_name, last_user_hobby
    global current_face_hist, last_face_seen_time

    features = extract_face_features(face_bgr)
    best_user = None

    with _users_lock:
        matrix = user_matrix
        snapshot = list(users)
        if matrix is not None and snapshot and matrix.shape[1] == features.size:
            scores = np.dot(matrix, features)
            idx = int(np.argmax(scores))
            if float(scores[idx]) >= MATCH_THRESHOLD:
                best_user = snapshot[idx]

    with _status_lock:
        current_face_hist = features.tolist()
        last_face_seen_time = time.time()
        if best_user is not None:
            last_seen_status = "KNOWN"
            last_user_name   = best_user["name"]
            last_user_hobby  = best_user.get("hobby", "Robotics")
        else:
            last_seen_status = "UNKNOWN"
            last_user_name   = ""
            last_user_hobby  = ""


# -------------------- Video stream --------------------
def _camera_off_jpeg():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(frame, "CAMERA OFF", (60, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 120), 2)
    ok, enc = cv2.imencode(".jpg", frame)
    return enc.tobytes() if ok else b""


def generate_frames():
    off_bytes = _camera_off_jpeg()
    counter = 0
    cached_box = None

    while True:
        mode = current_mode
        if mode == "OFF" or picam2 is None:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + off_bytes + b"\r\n")
            time.sleep(0.2)
            continue

        try:
            with _cam_lock:
                if current_mode == "OFF":
                    continue
                frame = picam2.capture_array()
        except Exception as exc:
            print(f"[CAM] Capture error: {exc}", flush=True)
            time.sleep(0.1)
            continue

        if frame is None or frame.size == 0:
            time.sleep(0.04)
            continue

        if frame.ndim == 3 and frame.shape[2] == 3 and app.config.get("PIXFMT") == "RGB":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if mode == "GREET" and not face_cascade.empty():
            counter += 1
            face_found_this_frame = False
            status_for_tick = "NONE"
            name_for_tick   = ""
            hobby_for_tick  = ""

            if counter % 3 == 0:
                small = cv2.resize(frame, (320, 240))
                gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.25, minNeighbors=4, minSize=(35, 35)
                )
                if len(faces):
                    sx, sy, sw, sh = max(faces, key=lambda b: b[2] * b[3])
                    x, y, w, h = sx * 2, sy * 2, sw * 2, sh * 2
                    x2 = min(x + w, frame.shape[1])
                    y2 = min(y + h, frame.shape[0])
                    roi = frame[max(y, 0):y2, max(x, 0):x2]
                    cached_box = (x, y, x2 - x, y2 - y)
                    face_found_this_frame = True
                    if roi.size:
                        match_face(roi)
                        with _status_lock:
                            status_for_tick = last_seen_status
                            name_for_tick   = last_user_name
                            hobby_for_tick  = last_user_hobby
                else:
                    cached_box = None
                    with _status_lock:
                        globals()["last_seen_status"]  = "NONE"
                        globals()["last_user_name"]    = ""
                        globals()["last_user_hobby"]   = ""
                        globals()["current_face_hist"] = None

            _greet_tick(face_found_this_frame or cached_box is not None,
                        status_for_tick, name_for_tick, hobby_for_tick)

            if cached_box is not None:
                x, y, w, h = cached_box
                with _status_lock:
                    status = last_seen_status
                    name   = last_user_name
                color = (0, 255, 0) if status == "KNOWN" else (0, 140, 255)
                label = name if status == "KNOWN" else "UNKNOWN"
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, max(20, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        ok, encoded = cv2.imencode(".jpg", frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, VIDEO_QUALITY])
        if ok:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + encoded.tobytes() + b"\r\n")


# -------------------- Background telemetry --------------------
_telemetry_cache = {"distance": None, "battery": None, "updated_at": 0.0}
_telemetry_lock = threading.Lock()


def _telemetry_reader():
    while True:
        if _i2c_busy.is_set():
            time.sleep(0.5)
            continue
        dist = None
        if sonar is not None and not _i2c_busy.is_set():
            try:
                v = sonar.read()
                if v is not None and float(v) > 0:
                    dist = round(float(v), 1)
            except Exception:
                pass
        bat = None
        if bat_adc is not None and not _i2c_busy.is_set():
            try:
                raw = bat_adc.read()
                if raw is not None and float(raw) > 0:
                    bat = round(float(raw) * 3.3 / 4095 * 3, 2)
            except Exception:
                pass
        with _telemetry_lock:
            _telemetry_cache["distance"] = dist
            _telemetry_cache["battery"] = bat
            _telemetry_cache["updated_at"] = time.time()
        time.sleep(0.5)


threading.Thread(target=_telemetry_reader, daemon=True).start()


# -------------------- Auth --------------------
def _check_token():
    if not API_TOKEN:
        return True
    token = request.args.get("token") or request.headers.get("X-Token", "")
    return token == API_TOKEN


def _api_guard():
    if not _check_token():
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    return None


# -------------------- Routes --------------------
@app.route("/")
def index():
    return render_template("cockpit.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/mode/walk_cam", methods=["POST"])
def toggle_walk_cam():
    guard = _api_guard()
    if guard:
        return guard
    try:
        target = "OFF" if current_mode == "WALK" else "WALK"
        set_camera_state(target)
        return jsonify({"mode": current_mode, "active": current_mode == "WALK"})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc),
                        "mode": current_mode}), 500


@app.route("/api/mode/greet_face", methods=["POST"])
def toggle_greet_face():
    """Single button: starts the full introduction cycle."""
    guard = _api_guard()
    if guard:
        return guard
    try:
        if current_mode == "GREET":
            set_camera_state("OFF")
            with _active_motion_lock:
                globals()["_active_motion"] = None
            _greet_set("IDLE", "")
            _i2c_busy.set()
            try:
                if crawler:
                    crawler.do_step("stand", 50)
            finally:
                _i2c_busy.clear()
            return jsonify({"mode": "OFF", "active": False})

        with _active_motion_lock:
            globals()["_active_motion"] = None

        _look_up()
        set_camera_state("GREET")
        _greet_set("LOOKING", "Looking for a face...")
        return jsonify({"mode": "GREET", "active": True})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc),
                        "mode": current_mode}), 500


@app.route("/api/telemetry")
def telemetry():
    guard = _api_guard()
    if guard:
        return guard

    with _telemetry_lock:
        distance = _telemetry_cache["distance"]
        battery  = _telemetry_cache["battery"]

    with _status_lock:
        status = last_seen_status
        name   = last_user_name
        hobby  = last_user_hobby
        f_time = last_face_seen_time

    if status != "NONE" and time.time() - f_time > FACE_TIMEOUT:
        status, name, hobby = "NONE", "", ""

    with _active_motion_lock:
        current_motion = _active_motion

    with _greet_lock:
        greet_state      = _greet["state"]
        greet_message    = _greet["message"]
        greet_message_id = _greet["message_id"]

    return jsonify({
        "distance":         distance,
        "battery":          battery,
        "ai_status":        status,
        "user_name":        name,
        "user_hobby":       hobby,
        "current_mode":     current_mode,
        "active_motion":    current_motion,
        "greet_state":      greet_state,
        "greet_message":    greet_message,
        "greet_message_id": greet_message_id,
    })


@app.route("/api/ai/save_profile", methods=["POST"])
def save_profile():
    guard = _api_guard()
    if guard:
        return guard

    if current_mode != "GREET":
        return jsonify({"status": "error",
                        "message": "Greet mode is not active"}), 400

    data  = request.get_json(silent=True) or {}
    name  = str(data.get("name", "")).strip()[:80]
    hobby = str(data.get("hobby", "Robotics")).strip()[:120] or "Robotics"

    with _status_lock:
        face_hist = list(current_face_hist) if current_face_hist else None
        face_time = last_face_seen_time

    if not name:
        return jsonify({"status": "error", "message": "Please enter a name"}), 400
    if (face_hist is None
            or len(face_hist) != FACE_FEATURE_LEN
            or time.time() - face_time > FACE_TIMEOUT):
        return jsonify({"status": "error",
                        "message": "No face detected - look at the camera"}), 400

    with _users_lock:
        backup = [dict(u) for u in users]
        existing = next((u for u in users if u["name"].casefold() == name.casefold()), None)
        record = {"name": name, "hobby": hobby, "hist": face_hist}
        if existing is None:
            users.append(record)
        else:
            existing.update(record)

        if not _save_users_locked():
            users.clear()
            users.extend(backup)
            _rebuild_matrix()
            return jsonify({"status": "error",
                            "message": "Could not write users.json"}), 500

    _greet_set("SAVED",
               f"Nice to meet you, {name}! Got it. "
               f"Your hobby is {hobby}.")
    return jsonify({"status": "ok", "name": name, "hobby": hobby})


@app.route("/api/move", methods=["POST"])
def handle_move():
    guard = _api_guard()
    if guard:
        return guard

    global _active_motion
    if crawler is None:
        return jsonify({"status": "error", "message": "Robot not available"}), 503

    data = request.get_json(silent=True) or {}
    action = str(data.get("action", "")).strip().lower()

    if action == "turn_left":
        action = "left"
    elif action == "turn_right":
        action = "right"

    if action in CONTINUOUS_ACTIONS:
        with _active_motion_lock:
            if _active_motion == action:
                _active_motion = None
                new_state = None
            else:
                _active_motion = action
                new_state = action
        return jsonify({
            "status": "ok", "action": action,
            "active": new_state == action,
            "active_motion": new_state,
        })

    if action in ONE_SHOT_ACTIONS:
        with _active_motion_lock:
            globals()["_active_motion"] = None

        def _oneshot():
            _i2c_busy.set()
            try:
                _run_step(action)
            except Exception as exc:
                print(f"[MOVE] {action} failed: {exc}", flush=True)
            finally:
                _i2c_busy.clear()

        threading.Thread(target=_oneshot, daemon=True).start()
        return jsonify({"status": "ok", "action": action,
                        "active": False, "active_motion": None})

    return jsonify({"status": "error",
                    "message": f"Unknown action: {action}"}), 400


@app.route("/api/health")
def health():
    return jsonify({
        "status":               "ok",
        "camera_available":     camera_available,
        "camera_mode":          current_mode,
        "face_cascade_loaded":  not face_cascade.empty(),
        "profiles":             len(users),
        "robot_available":      crawler is not None,
        "sonar_available":      sonar is not None,
        "adc_available":        bat_adc is not None,
        "auth_required":        bool(API_TOKEN),
    })


if __name__ == "__main__":
    print(f"[SERVER] http://{HOST}:{PORT}", flush=True)
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
