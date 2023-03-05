import tensorflow as tf
import cv2
import numpy as np
import json
import streamlit as st
import matplotlib.pyplot as plt

# Define image size and scale factor
image_height, image_width = 256, 256
scale_factor = 0.3

# Load metadata
with open('my-pose-model/metadata.json', 'r') as f:
    metadata = json.load(f)

# Define classes
if 'keypoints' in metadata:
    classes = metadata['keypoints']
else:
    classes = []
num_classes = len(classes)

# Define model architecture
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(image_width, image_height, 3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

# Load model weights
model.load_weights('C:/Users/sahil/PycharmProjects/Aap ki Avaj/my-pose-model/model.h5')

# Define webcam feed and pose detection
st.title('Pose Detection App')

# Initialize OpenCV webcam capture
cap = cv2.VideoCapture(0)

# Create a Streamlit canvas to show the webcam feed
canvas = st.image([], use_column_width=True)

# Define colors for skeleton
colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255]]

while True:
    # Capture frame from webcam
    ret, frame = cap.read()

    # Resize image
    resized_frame = cv2.resize(frame, (image_width, image_height))

    # Scale image values to [0,1]
    resized_frame = resized_frame / 255.0

    # Add batch dimension
    input_data = np.expand_dims(resized_frame, axis=0)

    # Make predictions with the model
    predictions = model.predict(input_data)

    # Get keypoint coordinates from predictions
    keypoint_coords = predictions[0]

    # Scale keypoint coordinates back to original image size
    keypoint_coords *= [image_width, image_height]

    # Draw skeleton on frame
    for i in range(num_classes):
        x, y = int(keypoint_coords[i][0]), int(keypoint_coords[i][1])
        cv2.circle(frame, (x, y), 5, colors[i], -1)

    # Show frame with skeleton on Streamlit canvas
    canvas.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close window
cap.release()
cv2.destroyAllWindows()
