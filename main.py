import cv2
import time
from pose_estimation import PoseEstimator

def main():
    estimator = PoseEstimator()
    cap = cv2.VideoCapture(0)
    prev_time = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
            
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process both pose and face
        pose_results = estimator.pose.process(rgb_frame)
        face_results = estimator.face_mesh.process(rgb_frame)
        
        if pose_results.pose_landmarks:
            # Pass both pose and face landmarks to draw
            estimator.draw_landmarks(
                frame, 
                pose_results.pose_landmarks,
                face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None
            )
            
            # Exercise analysis (unchanged)
            if estimator.current_exercise == "squat":
                estimator.analyze_squat(pose_results.pose_landmarks.landmark)
            elif estimator.current_exercise == "pushup":
                estimator.analyze_pushup(pose_results.pose_landmarks.landmark)
        
        # Display info (updated layout)
        cv2.putText(frame, f"Exercise: {estimator.current_exercise.upper()}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Reps: {estimator.rep_count}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # FPS calculation (unchanged)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1]-100, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('AI Fitness Trainer', frame)
        
        key = cv2.waitKey(5)
        if key == ord('q'):
            break
        elif key == ord('1'):
            estimator.change_exercise("squat")
        elif key == ord('2'):
            estimator.change_exercise("pushup")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()