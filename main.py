import cv2
from selenium.webdriver.common.by import By
from selenium import webdriver
import face_recognition
import numpy as np

#saves an image of the webpage provided
def scrapeSite(url, imgName = "scrape.png"):
    driver = webdriver.Firefox()
    driver.get(url)
    driver.save_full_page_screenshot(imgName)
    driver.quit()
    return imgName



#replace url with site to search for faces
image = scrapeSite("https://en.wikipedia.org/wiki/Tyler,_the_Creator")



facesDB = cv2.imread(image)

knownFaceLocations = face_recognition.face_locations(facesDB)
knownFaceEncodings = face_recognition.face_encodings(facesDB, knownFaceLocations)
#print(f"{len(knownFaceEncodings)} faces found")
frameCount = 0

cap = cv2.VideoCapture(0)

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
            #use the if statement instead if you only want to provide likely face matches, currently provides best match even at low confidence
            """
            if matches[best_match_index]:
                location = knownFaceLocations[best_match_index]
                distance = (1-faceDistances[best_match_index])*100
            """
            location = knownFaceLocations[best_match_index]
            distance = (1-faceDistances[best_match_index])*100

    #process once every 5 frames, change mod to change process interval
    frameCount = (frameCount+1)%5

    if faceLocations:

        (top, right, bottom, left) = faceLocations[0]

        w = right-left
        h = bottom-top

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        font = cv2.FONT_HERSHEY_TRIPLEX
        #provide closeness of faces in box around user's face
        cv2.putText(frame, f"{round(distance, 2)} %", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        cv2.rectangle(frame, (0,0), (w, h), (0, 0, 255), 2)
        if distance:
            #find closest face and display at top left of window
            (topC, rightC, bottomC, leftC) = knownFaceLocations[best_match_index]
            crop = facesDB[topC:bottomC, leftC:rightC]
            crop = cv2.resize(crop, (w, h))
            frame[0:h, 0:w] = crop
        else:
            cv2.putText(frame, "Face not found", (2, int(h/2)), font, 0.5, (255, 255, 255), 1)


    cv2.imshow("frame", frame)
    keyPress = cv2.waitKey(1)
    #go until esc is pressed
    if keyPress == 27:
        break

cap.release()
cv2.destroyAllWindows()

