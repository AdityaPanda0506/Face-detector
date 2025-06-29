import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp
from ultralytics import YOLO

# Initialize YOLO and FaceMesh
yolo_model = YOLO("yolov8n.pt")
mp_face = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Thresholds
EYE_OPEN_THRESH = 0.2
GAZE_FORWARD_THRESH_X = 0.1
GAZE_FORWARD_THRESH_Y = 0.1
MAX_HISTORY = 15
emotion_history = []
score_history = []

def eye_openness_ratio(iris, eye):
    l = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    w = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return l / w

def estimate_gaze(left_eye, right_eye, left_iris, right_iris):
    le_c = np.mean(left_eye, axis=0)
    re_c = np.mean(right_eye, axis=0)
    li_c = np.mean(left_iris, axis=0)
    ri_c = np.mean(right_iris, axis=0)
    gx = ((li_c[0] - le_c[0]) + (ri_c[0] - re_c[0])) / (le_c[0] * 2)
    gy = ((li_c[1] - le_c[1]) + (ri_c[1] - re_c[1])) / (le_c[1] * 2)
    return gx, gy

def calculate_score(emotion, eye_ratio, gaze_x, gaze_y, phone_detected):
    weights = {
        'neutral': 1.0, 'happy': 1.2, 'surprise': 0.8,
        'angry': 0.3, 'fear': 0.3, 'sad': 0.2, 'disgust': 0.2
    }
    score = weights.get(emotion, 0.5)
    score += max(0.0, min((eye_ratio - EYE_OPEN_THRESH) * 2, 1.0))
    gaze_score = 1.0 if abs(gaze_x) < GAZE_FORWARD_THRESH_X and abs(gaze_y) < GAZE_FORWARD_THRESH_Y else -0.5
    score += gaze_score
    if phone_detected:
        score -= 1.0
    return round(max(0.0, min(score * 2, 10.0)), 2)

def get_learning_state(score):
    if score >= 9: return "Engaged"
    elif 7 <= score < 9: return "Distracted"
    elif 5 <= score < 7: return "Confused"
    elif 3 <= score < 5: return "Tired"
    else: return "Frustrated"

def get_feedback(state):
    """
    Returns structured feedback including message, action type, resource link, and pause flag.
    
    Args:
        state (str): Learning state (Engaged, Distracted, etc.)
        
    Returns:
        dict: {
            'message': str,
            'action_type': str,
            'resource': str,
            'should_pause': bool
        }
    """
    if state == "Engaged":
        return {
            "message": "Great focus! Keep it up.",
            "action_type": "continue",
            "resource": "",
            "should_pause": False
        }

    elif state == "Distracted":
        return {
            "message": "You seem slightly distracted. Want to try a mini quiz?",
            "action_type": "quiz",
            "resource": "/mini-quiz",
            "should_pause": False
        }

    elif state == "Confused":
        return {
            "message": "Looks like you're struggling. Would you like help?",
            "action_type": "help",
            "resource": "/help",
            "should_pause": True
        }

    elif state == "Tired":
        return {
            "message": "You might be getting tired. Consider taking a short break.",
            "action_type": "break",
            "resource": "/break",
            "should_pause": True
        }

    elif state == "Frustrated":
        return {
            "message": "It seems like you're frustrated. Let's pause and regroup.",
            "action_type": "pause",
            "resource": "/pause",
            "should_pause": True
        }

    else:  # Unknown or fallback
        return {
            "message": "Good job staying focused.",
            "action_type": "continue",
            "resource": "",
            "should_pause": False
        }

def analyze_frame(frame):
    global emotion_history, score_history

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face.process(rgb_frame)

    # Phone detection using YOLO
    phone_detected = False
    detections = yolo_model(frame)[0]
    for box in detections.boxes:
        cls = int(box.cls[0])
        label = yolo_model.names[cls]
        if label == "cell phone" and float(box.conf[0]) > 0.4:
            phone_detected = True
            break

    if not results.multi_face_landmarks:
        return {
            'emotion': 'No face',
            'score': 0.0,
            'state': 'Unknown',
            'feedback': 'No face detected',
            'phone': phone_detected
        }

    mesh = results.multi_face_landmarks[0].landmark
    pts = [(int(p.x * w), int(p.y * h)) for p in mesh]
    le = [pts[i] for i in [33, 133, 159, 145, 153, 154]]
    re = [pts[i] for i in [362, 263, 386, 374, 380, 385]]
    li = [pts[i] for i in [468, 469, 470, 471]]
    ri = [pts[i] for i in [473, 474, 475, 476]]

    eye_ratio = (eye_openness_ratio(li, le) + eye_openness_ratio(ri, re)) / 2
    gaze_x, gaze_y = estimate_gaze(le, re, li, ri)

    try:
        det = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
        if isinstance(det, list): det = det[0]
        emotion = det['dominant_emotion']
    except Exception:
        emotion = 'neutral'

    score = calculate_score(emotion, eye_ratio, gaze_x, gaze_y, phone_detected)
    emotion_history.append(emotion)
    score_history.append(score)
    if len(emotion_history) > MAX_HISTORY: emotion_history.pop(0)
    if len(score_history) > MAX_HISTORY: score_history.pop(0)

    avg_score = sum(score_history) / len(score_history)
    dominant_emotion = max(set(emotion_history), key=emotion_history.count)
    state = get_learning_state(avg_score)
    feedback = get_feedback(state)

    return {
        'emotion': dominant_emotion,
        'score': round(avg_score / 1.1, 2),
        'state': state,
        'feedback': feedback,
        'phone': phone_detected
    }