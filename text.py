import cv2
import streamlit as st
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyautogui

pyautogui.FAILSAFE = False

st.markdown("<h1 style='color:red; background-color: lightcoral;text-align: center; color: black;'>Welcome to Aap ki Avaj!...</h1>", unsafe_allow_html=True)

st.markdown(" ", unsafe_allow_html=True)

st.title("Sign Language Recognition")

# Add some introductory text
st.write("Welcome to Aap ki Avaj This web application allows you to communicate using Indian Sign Language, "
         "which is used by millions of people in India who are deaf or hard-of-hearing. By using this application, you can "
         "type messages that will be displayed in sign language or use your hands to create gestures that will be "
         "translated into text messages. This application is meant to bridge the communication gap between those who know "
         "sign language and those who do not, making communication easier and more accessible for everyone.")

# Add a streamlit video display for the webcam feed
video_display = st.empty()

# Load the hand detector and classification models
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300
labels = ["A", "B", "C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","THANK U","HELLO","Spider Man"," "]
labels2 = ["Where is this address?",
           "Where is the bus stop?",
           "Can you recommend a good coffee shop?",
           "Do you know where the nearest drugstore is?",
           "Excuse me, could you tell me where the elevator is?",
           "Can you help me find a good fish market?",
           "Could you give me directions to the grocery store?",
           "Can you recommend a good hotel nearby?",
           "Is there a good Italian restaurant in the area?",
           "Do you know where the nearest Japanese restaurant is?",
           "Can you tell me where the nearest kindergarten is?",
           "Where is the nearest library?",
           "Can you recommend a good Mexican restaurant?",
           "Do you know where the nearest newsstand is?",
           "Where is the nearest office supply store?",
           "Do you know where the nearest post office is?",
           "Are there any good quality restaurants in the area?",
           "Where is the nearest railway station?",
           "Can you recommend a good seafood restaurant?",
           "Where is the nearest taxi stand?",
           "Do you know where the nearest university is?",
           "Can you tell me where the nearest veterinarian is?",
           "Where is the nearest wine shop?",
           "I'm not sure what to ask for X, sorry!",
           "Do you know where the nearest yoga studio is?",
           "Can you recommend a good zoo in the area?",
           "Thank you for your help",
           "HELLO",
           "Spider Man",
           " "]

sahil=1
start_button = st.button("Start")
stop_button = st.button("Stop")
Refresh = st.button("Refresh")
if Refresh:
    sahil=1
    start_button=True
    Refresh=False

while True:
    if start_button:
        success, img = cap.read()
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Add boundary checks to ensure hand is within the frame
            if x - offset >= 0 and y - offset >= 0 and x + w + offset < img.shape[1] and y + h + offset < img.shape[0]:
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                imgCropShape = imgCrop.shape

                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    print(prediction, index)

                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)

                if prediction == "L":
                    pyautogui.press('space')
                elif prediction == "V":
                    pyautogui.press('v')

                cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                              (x + w + offset, y - offset - 50 + 50), (255, 255, 255), cv2.FILLED)
                if index >= 0 and index < len(labels):
                    cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset),
                              (x + w + offset, y + h + offset), (255, 255, 255), 4)

                if sahil==1:
                    new_text = labels2[index]
                    text_input = st.text_area("",value=new_text)
                    sahil=0

    if stop_button:
        # Break out of the loop
        break

    # Display the video in the stframe
    video_display.image(imgOutput, channels="BGR")

# Add some information on speech impairments
st.header("Communication and Disabilities")
st.write("The ability to communicate is essential for everyone, but for people with disabilities, communication can be a "
         "challenge. In India, the 2011 Census recorded a total of 26.8 million people with disabilities, of which 7% had "
         "speech impairments. Technologies that enable communication, such as Indian Sign Language recognition, can help "
         "bridge the communication gap and make communication more accessible for everyone.")

st.image("img/AKA.jpeg")


# Add some information on Indian Sign Language
st.header("What is Indian Sign Language?")
st.write("Indian Sign Language (ISL) is the sign language used by millions of people in India who are deaf or hard-of-hearing. "
         "ISL has its own unique grammar and syntax, and uses a combination of hand gestures, facial expressions, and body "
         "language to convey meaning. Just like spoken languages, ISL also has regional variations and dialects.")

