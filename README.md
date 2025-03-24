# AI Fitness Trainer with Pose Estimation


A real-time fitness trainer application that uses **MediaPipe's Pose Estimation** and **Face Mesh** to:
- Track exercise form (squats, push-ups)
- Count repetitions
- Detect body orientation (front, side, back views)
- Provide visual feedback

## Features

- **Exercise Tracking**:
  - Squat detection with rep counting
  - Push-up detection with rep counting
- **View Detection**:
  - Front, left side, right side, and back view detection
  - Uses facial landmarks for orientation analysis
- **Real-time Feedback**:
  - Visual landmark display
  - Exercise stage detection
  - FPS counter for performance monitoring

## Technologies Used

- Python 3.x
- OpenCV (cv2) for video processing
- MediaPipe for pose and face landmark detection
- NumPy for angle calculations

## Installation

1. Clone the repository:
   ```bash
   git clone 
   cd ai-fitness-trainer