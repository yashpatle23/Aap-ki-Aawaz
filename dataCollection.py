import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

offset = 10
imgSize = 300

folder = "Data/Z"
counter = 0

while counter<100:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand1, hand2 = hands[0], hands[1] if len(hands) > 1 else None

        # Check the position of the hands detected (always detect left hand first)
        if hand2 and hand2["bbox"][0] < hand1["bbox"][0]:
            hand1, hand2 = hand2, hand1

        x1, y1, w1, h1 = hand1['bbox']
        imgWhite = np.ones((imgSize, imgSize * 2, 3), np.uint8) * 255

        # Add boundary checks to ensure hand1 is within the frame
        if x1 - offset >= 0 and y1 - offset >= 0 and x1 + w1 + offset < img.shape[1] and y1 + h1 + offset < img.shape[0]:
            imgCrop1 = img[y1 - offset:y1 + h1 + offset, x1 - offset:x1 + w1 + offset]

            imgCropShape = imgCrop1.shape

            aspectRatio1 = h1 / w1

            if aspectRatio1 > 1:
                k = imgSize / h1
                wCal1 = math.ceil(k * w1)
                imgResize1 = cv2.resize(imgCrop1, (wCal1, imgSize))
                imgResizeShape = imgResize1.shape
                wGap1 = math.ceil((imgSize - wCal1) / 2)
                imgWhite[:, wGap1:wCal1 + wGap1] = imgResize1

            else:
                k = imgSize / w1
                hCal1 = math.ceil(k * h1)
                imgResize1 = cv2.resize(imgCrop1, (imgSize, hCal1))
                imgResizeShape = imgResize1.shape
                hGap1 = math.ceil((imgSize - hCal1) / 2)
                imgWhite[hGap1:hCal1 + hGap1, :imgSize] = imgResize1

        if hand2:
            x2, y2, w2, h2 = hand2['bbox']
            # Add boundary checks to ensure hand2 is within the frame
            if x2 - offset >= 0 and y2 - offset >= 0 and x2 + w2 + offset < img.shape[1] and y2 + h2 + offset < img.shape[0]:
                imgCrop2 = img[y2 - offset:y2 + h2 + offset, x2 - offset:x2 + w2 + offset]

                imgCropShape = imgCrop2.shape

                aspectRatio2 = h2 / w2

                if aspectRatio2 > 1:
                    k = imgSize / h2
                    wCal2 = math.ceil(k * w2)
                    imgResize2 = cv2.resize(imgCrop2, (wCal2, imgSize))
                    imgResizeShape = imgResize2.shape
                    wGap2 = math.ceil((imgSize - wCal2) / 2)
                    imgWhite[:, imgSize + wGap2:wCal2 + imgSize + wGap2] = imgResize2

                else:
                    k = imgSize / w2
                    hCal2 = math.ceil(k * h2)
                    imgResize2 = cv2.resize(imgCrop2, (imgSize, hCal2))
                    imgResizeShape = imgResize2.shape
                    hGap = math.ceil((imgSize - hCal2) / 2)
                    imgWhite[hGap:hCal2 + hGap, imgSize:] = imgResize2

        cv2.imshow("ImageWhite", imgWhite)

        img1 = cv2.flip(img, 1)
        cv2.imshow("Image", img1)
        key = cv2.waitKey(15)
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)
