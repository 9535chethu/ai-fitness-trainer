# import cv2
# import mediapipe as mp
# import time
# import math

# # Initialize MediaPipe solutions
# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose

# def determine_pose_view(landmarks):
#     """Enhanced view detection with visibility checks"""
#     try:
#         # Get key landmarks with visibility checks
#         nose = landmarks.landmark[mp_pose.PoseLandmark.NOSE]
#         left_ear = landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]
#         right_ear = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]
#         left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
#         right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

#         # Calculate visibility scores (0-1)
#         nose_visible = nose.visibility > 0.5
#         left_ear_visible = left_ear.visibility > 0.5
#         right_ear_visible = right_ear.visibility > 0.5
#         left_shoulder_visible = left_shoulder.visibility > 0.7
#         right_shoulder_visible = right_shoulder.visibility > 0.7

#         # Rear view detection (no nose visible, ears visible)
#         if not nose_visible and (left_ear_visible or right_ear_visible):
#             return "REAR VIEW", (0, 165, 255)  # Orange

#         # Front view detection (nose centered between shoulders)
#         if nose_visible and left_shoulder_visible and right_shoulder_visible:
#             shoulder_midpoint = (left_shoulder.x + right_shoulder.x) / 2
#             if abs(nose.x - shoulder_midpoint) < 0.1:  # Nose is centered
#                 return "FRONT VIEW", (0, 255, 255)  # Yellow

#         # Side view detection (one shoulder much more visible)
#         shoulder_visibility_diff = abs(left_shoulder.visibility - right_shoulder.visibility)
#         if shoulder_visibility_diff > 0.3:
#             if left_shoulder.visibility > right_shoulder.visibility:
#                 return "RIGHT SIDE VIEW", (255, 0, 0)  # Red
#             else:
#                 return "LEFT SIDE VIEW", (255, 0, 0)  # Red

#         # 3/4 view detection (partial visibility)
#         if nose_visible and (left_ear_visible or right_ear_visible):
#             return "3/4 VIEW", (255, 0, 255)  # Purple

#         return "VIEW UNCERTAIN", (0, 0, 255)  # Red

#     except Exception as e:
#         print(f"View detection error: {e}")
#         return "DETECTION ERROR", (0, 0, 255)

# def draw_landmarks(image, landmarks):
#     """Draw pose landmarks with custom styling"""
#     mp_drawing.draw_landmarks(
#         image,
#         landmarks,
#         mp_pose.POSE_CONNECTIONS,
#         landmark_drawing_spec=mp_drawing.DrawingSpec(
#             color=(0, 255, 0), thickness=2, circle_radius=2),
#         connection_drawing_spec=mp_drawing.DrawingSpec(
#             color=(255, 0, 0), thickness=2))
    
#     # Highlight key points for view detection
#     for landmark in [mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_EAR, 
#                     mp_pose.PoseLandmark.RIGHT_EAR, mp_pose.PoseLandmark.LEFT_SHOULDER,
#                     mp_pose.PoseLandmark.RIGHT_SHOULDER]:
#         if landmarks.landmark[landmark].visibility > 0.5:
#             x = int(landmarks.landmark[landmark].x * image.shape[1])
#             y = int(landmarks.landmark[landmark].y * image.shape[0])
#             cv2.circle(image, (x, y), 5, (255, 255, 255), -1)

# def calculate_angle(a, b, c):
#     """Calculate angle between three points"""
#     a = np.array(a)
#     b = np.array(b)
#     c = np.array(c)
    
#     radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
#     angle = np.abs(radians*180.0/np.pi)
    
#     if angle > 180.0:
#         angle = 360 - angle
        
#     return angle

# def main():
#     # Initialize MediaPipe Pose with optimized parameters
#     pose = mp_pose.Pose(
#         static_image_mode=False,
#         model_complexity=1,
#         smooth_landmarks=True,
#         enable_segmentation=False,
#         min_detection_confidence=0.7,
#         min_tracking_confidence=0.7
#     )
    
#     # Initialize webcam
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return
    
#     # For FPS calculation
#     prev_time = 0
#     fps = 0
    
#     try:
#         while cap.isOpened():
#             # Read frame
#             success, frame = cap.read()
#             if not success:
#                 print("Ignoring empty camera frame.")
#                 continue
            
#             # Convert to RGB and process
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = pose.process(frame_rgb)
            
#             # Draw landmarks if detected
#             if results.pose_landmarks:
#                 draw_landmarks(frame, results.pose_landmarks)
                
#                 # Determine and display view
#                 view_text, color = determine_pose_view(results.pose_landmarks)
#                 cv2.putText(frame, view_text, (10, 70),
#                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
#             # Calculate and display FPS
#             curr_time = time.time()
#             fps = 0.9 * fps + 0.1 * (1 / (curr_time - prev_time))  # Smoothed FPS
#             prev_time = curr_time
#             cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
#                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
#             # Display frame
#             cv2.imshow('Pose Estimation', frame)
            
#             # Exit on 'q' key
#             if cv2.waitKey(5) & 0xFF == ord('q'):
#                 break
                
#     except KeyboardInterrupt:
#         print("Program stopped by user")
#     finally:
#         # Clean up
#         cap.release()
#         cv2.destroyAllWindows()
#         pose.close()

# if __name__ == "__main__":
#     import numpy as np
#     main()


import cv2
import mediapipe as mp
import numpy as np
import time

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

