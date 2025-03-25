import cv2
import mediapipe as mp
import numpy as np

class PoseEstimator:
    def __init__(self):
        self.rep_count = 0
        self.stage = None
        self.current_exercise = "squat"
        self.current_view = "unknown"
        
        # Initialize pose and face detection
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.draw_utils = mp.solutions.drawing_utils
        self.drawing_spec = mp.solutions.drawing_utils.DrawingSpec(
            color=(0, 255, 0), thickness=2, circle_radius=2)

    def determine_view(self, face_landmarks):
        """Detect front/left/right/back view based on facial features"""
        try:
            # Key facial landmarks indices
            NOSE_TIP = 1
            LEFT_EYE = 33
            RIGHT_EYE = 263
            MOUTH_LEFT = 61
            MOUTH_RIGHT = 291
            
            # Get visibility of key points
            nose = face_landmarks.landmark[NOSE_TIP]
            left_eye = face_landmarks.landmark[LEFT_EYE]
            right_eye = face_landmarks.landmark[RIGHT_EYE]
            mouth_left = face_landmarks.landmark[MOUTH_LEFT]
            mouth_right = face_landmarks.landmark[MOUTH_RIGHT]
            
            # Check visibility (z-value for depth)
            nose_visible = nose.z < 0.1  # Nose is closer when visible
            left_eye_visible = left_eye.z < 0.1
            right_eye_visible = right_eye.z < 0.1
            mouth_visible = (mouth_left.z < 0.1) or (mouth_right.z < 0.1)
            
            # View detection logic
            if nose_visible and (left_eye_visible or right_eye_visible) and mouth_visible:
                self.current_view = "FRONT VIEW"
            elif left_eye_visible and not right_eye_visible:
                self.current_view = "RIGHT SIDE"  # Left eye visible = right side view
            elif right_eye_visible and not left_eye_visible:
                self.current_view = "LEFT SIDE"   # Right eye visible = left side view
            elif not (nose_visible or left_eye_visible or right_eye_visible):
                self.current_view = "BACK VIEW"
            else:
                self.current_view = "UNKNOWN"
                
        except Exception as e:
            print(f"View detection error: {e}")
            self.current_view = "UNKNOWN"

    def draw_landmarks(self, image, landmarks, face_landmarks=None):
        """Draw pose landmarks and view information"""
        # Draw pose landmarks
        self.draw_utils.draw_landmarks(
            image, landmarks, mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.drawing_spec,
            connection_drawing_spec=self.drawing_spec
        )
        
        # Determine and display view if face landmarks available
        if face_landmarks:
            self.determine_view(face_landmarks)
            # Display view information
            cv2.putText(image, f"View: {self.current_view}", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 0), 2)  # Yellow text

    @staticmethod
    def calculate_angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        return angle if angle <= 180 else 360 - angle

    def analyze_squat(self, landmarks):
        hip = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x, 
               landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].x, 
                landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].x, 
                 landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].y]
        
        angle = self.calculate_angle(hip, knee, ankle)
        
        if angle > 160:
            self.stage = "up"
        if angle < 90 and self.stage == "up":
            self.stage = "down"
            self.rep_count += 1

    def analyze_pushup(self, landmarks):
        shoulder = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                   landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].x, 
                 landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].x, 
                 landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].y]
        
        angle = self.calculate_angle(shoulder, elbow, wrist)
        
        if angle > 160:
            self.stage = "up"
        if angle < 90 and self.stage == "up":
            self.stage = "down"
            self.rep_count += 1

    def change_exercise(self, exercise):
        self.current_exercise = exercise
        self.rep_count = 0
        self.stage = None

