# AI Attendance System using Face Recognition

An intelligent attendance management system that uses real-time face recognition to automatically mark student attendance via webcam.

## Features

- Real-time face detection and recognition using webcam
- Automatic attendance marking with timestamp
- Saves attendance records to CSV file
- Supports multiple students
- Unknown face detection (shown in red box)
- Known face detection (shown in green box)

## Technologies Used

- Python 3.12
- OpenCV - for webcam and image processing
- face_recognition - for AI face detection and recognition
- dlib - for facial landmark detection
- NumPy - for numerical operations
- Pandas - for data handling

## Project Structure

```
AI-Attendance-System/
├── images/
│     ├── Student1.jpg
│     ├── Student2.jpg
│     └── Student3.jpg
├── attendance.csv
├── main.py
└── README.md
```

## How It Works

1. The system loads face photos from the `images/` folder
2. It encodes each face using AI face recognition models
3. Webcam opens and detects faces in real time
4. Each detected face is compared with stored encodings
5. If a match is found, the student name and time is saved to `attendance.csv`
6. Unknown faces are shown with a red box

## Installation

1. Clone the repository
```
git clone https://github.com/Agent1810/AI-Attendance-System.git
cd AI-Attendance-System
```

2. Install required libraries
```
pip install opencv-python
pip install numpy==1.26.4
pip install dlib
pip install face-recognition
pip install face-recognition-models
```

3. Add student photos to the `images/` folder
   - Name each photo as the student's name (e.g. `John.jpg`)
   - Use clear front-facing photos with only one face

4. Run the program
```
python main.py
```

5. Press **Enter** to stop the webcam

## Output

Attendance is saved in `attendance.csv` with the following format:

```
Name, Time
MONISH, 10:30:25
MITHUNRAJ, 10:31:10
```

## Notes

- Make sure each photo in the `images/` folder contains only one face
- Good lighting improves recognition accuracy
- Tolerance is set to 0.45 for accurate matching

## Author

- GitHub: [@Agent1810](https://github.com/Agent1810)
