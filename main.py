import cv2
from selenium.webdriver.common.by import By
from selenium import webdriver
import face_recognition
import numpy as np


def scrapeSite(url, imgName = "scrape.png"):
    driver = webdriver.Firefox()
    driver.get(url)
    driver.save_full_page_screenshot(imgName)
    driver.quit()
    return imgName

"""
cap = cv2.VideoCapture(0)
img = cv2.imread(scrapeSite("https://en.wikipedia.org/wiki/Marisa_Tomei"))

grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


face = face_classifier.detectMultiScale(grayscale, 1.1, 5, minSize=(40,40))

print(len(face))

for (x, y, w, h) in face:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

import matplotlib.pyplot as plt

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(20,10))
plt.imshow(img_rgb)
plt.axis('off')

print("finished")



while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture")
        break

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(grayscale, 1.1, 5, minSize=(40,40))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 4)

    cv2.imshow("frame", frame)
    keyPress = cv2.waitKey(1)
    
    if keyPress == 27:
        break




cap.release()
cv2.destroyAllWindows()

"""


image = scrapeSite("https://en.wikipedia.org/wiki/Tyler,_the_Creator")

facesDB = cv2.imread(image)

print("image scraped")
knownFaceLocations = face_recognition.face_locations(facesDB)
knownFaceEncodings = face_recognition.face_encodings(facesDB, knownFaceLocations)
print(f"{len(knownFaceEncodings)} faces found")
frameCount = 0

cap = cv2.VideoCapture(0)

print("opening camera")

while True:
    ret, frame = cap.read()
    if not frameCount:
        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        faceLocations = face_recognition.face_locations(rgbFrame)
        faceEncodings = face_recognition.face_encodings(rgbFrame, faceLocations)
        location = "Unknown"
        distance = 1
        if faceEncodings:
            face = faceEncodings[0]
            matches = face_recognition.compare_faces(knownFaceEncodings, face)
            faceDistances = face_recognition.face_distance(knownFaceEncodings, face)
            best_match_index = np.argmin(faceDistances)
            """
            if matches[best_match_index]:
                location = knownFaceLocations[best_match_index]
                distance = faceDistances[best_match_index]
            """
            location = knownFaceLocations[best_match_index]
            distance = 1-faceDistances[best_match_index]

        
    frameCount = (frameCount+1)%5

    if faceLocations:

        (top, right, bottom, left) = faceLocations[0]

        w = right-left
        h = bottom-top

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        font = cv2.FONT_HERSHEY_TRIPLEX
        cv2.putText(frame, f"{round(distance, 2)} %", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        cv2.rectangle(frame, (0,0), (w, h), (0, 0, 255), 2)
        if distance:
            (topC, rightC, bottomC, leftC) = knownFaceLocations[best_match_index]
            print((topC, rightC, bottomC, leftC))
            
            print([leftC,rightC, topC,bottomC])

            crop = facesDB[topC:bottomC, leftC:rightC]
            cv2.imshow("cropped face", crop)

            crop = cv2.resize(crop, (w, h))
            frame[0:h, 0:w] = crop
        else:
            cv2.putText(frame, "face not found", (2, int(h/2)), font, 0.5, (255, 255, 255), 1)


    cv2.imshow("frame", frame)
    keyPress = cv2.waitKey(1)
    
    if keyPress == 27:
        break

cv2.waitKey(1000)

cap.release()
cv2.destroyAllWindows()

