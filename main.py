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



facesDB = face_recognition.load_image_file(scrapeSite("https://en.wikipedia.org/wiki/Marisa_Tomei"))
print("image scraped")
knownFaceLocations = face_recognition.face_locations(facesDB)
knownFaceEncodings = face_recognition.face_encodings(facesDB, knownFaceLocations)
print("faces found")
frameCount = 0

cap = cv2.VideoCapture(0)

print("opening camera")

while True:
    ret, frame = cap.read()
    if not frameCount:
        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        faceLocations = face_recognition.face_locations(rgbFrame)
        faceEncodings = face_recognition.face_encodings(rgbFrame, faceLocations)

        locations = []
        distances = []
        for face in faceEncodings:
            matches = face_recognition.compare_faces(knownFaceEncodings, face)
            location = "Unknown"
            distance = 0
            faceDistances = face_recognition.face_distance(knownFaceEncodings, face)
            best_match_index = np.argmin(faceDistances)
            print(type(faceDistances[best_match_index]))
            if matches[best_match_index]:
                location = knownFaceLocations[best_match_index]
                distance = faceDistances[best_match_index]

            distances.append(distance)
            locations.append(location)
        
    frameCount = (frameCount+1)%5

    for (top, right, bottom, left), confidence in zip(faceLocations, distances):
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, str(confidence), (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


    cv2.imshow("frame", frame)
    print("frame processed")
    keyPress = cv2.waitKey(1)
    
    if keyPress == 27:
        break

cap.release()
cv2.destroyAllWindows()

