import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'images'
images = []
classNames = []

myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    if curImg is None:
        print(f"Could not read image: {cl}, skipping...")
        continue
    curImg = cv2.cvtColor(curImg, cv2.COLOR_BGR2RGB)
    curImg = np.ascontiguousarray(curImg, dtype=np.uint8)
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

print("Loaded students:", classNames)

def findEncodings(images):
    encodeList = []
    for i, img in enumerate(images):
        img = np.ascontiguousarray(img, dtype=np.uint8)
        encodings = face_recognition.face_encodings(img)
        if len(encodings) == 0:
            print(f"No face found in image {classNames[i]}, skipping...")
            continue
        encodeList.append(encodings[0])
        print(f"Encoded: {classNames[i]}")
    return encodeList

encodeListKnown = findEncodings(images)
print("Encoding Complete —", len(encodeListKnown), "faces loaded")

def markAttendance(name):
    with open('attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

cap = cv2.VideoCapture(0)

TOLERANCE = 0.45  # Lower = stricter matching (0.4 to 0.5 recommended)

while True:
    success, img = cap.read()
    if not success:
        print("Could not read from webcam.")
        break

    imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)
    imgSmall = np.ascontiguousarray(imgSmall, dtype=np.uint8)

    facesCurFrame = face_recognition.face_locations(imgSmall)
    encodesCurFrame = face_recognition.face_encodings(imgSmall, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        # Only accept match if distance is below tolerance
        if faceDis[matchIndex] < TOLERANCE:
            name = classNames[matchIndex].upper()
            markAttendance(name)
            color = (0, 255, 0)  # Green box for known face
        else:
            name = "UNKNOWN"
            color = (0, 0, 255)  # Red box for unknown face

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
