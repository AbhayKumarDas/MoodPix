import cv2
from keras._tf_keras.keras.models import model_from_json
import numpy as np
import time
import requests
import io
from PIL import Image

# Load the emotion detection model
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

# Load the Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to extract features from the image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Emotion labels and corresponding colors
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
colors = {
    'neutral': 'green',
    'happy': 'yellow',
    'sad': 'blue',
    'surprise': 'orange',
    'angry': 'red',
    'fear': 'purple',
    'disgust': 'brown'
}

# Hugging Face API configuration
API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
headers = {"Authorization": "Bearer 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"}  # My actual API key

# Function to query the Hugging Face model
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    
    # Check if response is an image
    if response.status_code == 200 and 'image' in response.headers['Content-Type']:
        return response.content
    else:
        print("Error: The response is not an image or API request failed.")
        print(f"Status Code: {response.status_code}")
        print(f"Response Content: {response.content}")
        return None

# Initialize the webcam
webcam = cv2.VideoCapture(0)

# Time tracking for emotion consistency
emotion_time = {}
current_emotion = None

while True:
    ret, im = webcam.read()

    if not ret or im is None:
        print("Failed to capture frame from webcam. Exiting...")
        break
    
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(im, 1.3, 5)
    
    try:
        detected_emotion = None
        
        for (p, q, r, s) in faces:
            image = gray[q:q+s, p:p+r]
            cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
            image = cv2.resize(image, (48, 48))
            img = extract_features(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            
            # Display the emotion label on the image
            cv2.putText(im, '%s' % (prediction_label), (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
            
            # Check for emotion consistency
            if prediction_label == current_emotion:
                if time.time() - emotion_time.get('start_time', time.time()) > 5:
                    # Stop the webcam
                    webcam.release()
                    cv2.destroyAllWindows()
                    
                    # Generate character description
                    color = colors[current_emotion]
                    description = (f"A 3D cartoonish character with a {current_emotion} expression, featuring exaggerated facial "
                                   f"features that capture the depth of this emotion. The entire scene, from the character's "
                                   f"clothing to the background, is immersed in various shades of {color}, creating a cohesive "
                                   f"and visually striking atmosphere that fully embodies the {current_emotion}. The lighting "
                                   f"and shadows within the scene emphasize the {current_emotion}, making the character's mood "
                                   f"vividly come to life in a {color}-toned world.")
                    print()
                    print(f"The detected emotion is {current_emotion}.")
                    print()
                    print(description)
                    print()
                    
                    # Use the description as a prompt for the text-to-image model
                    image_bytes = query({"inputs": description})
                    
                    if image_bytes:
                        # Open and display the generated image
                        image = Image.open(io.BytesIO(image_bytes))
                        image.show()  # This will open the image using the default image viewer
                    else:
                        print("Failed to generate the image.")
                    
                    break
            else:
                current_emotion = prediction_label
                emotion_time['start_time'] = time.time()
                
        cv2.imshow("Output", im)
        if cv2.waitKey(27) & 0xFF == 27:  # Exit loop if 'Esc' key is pressed
            break

    except cv2.error as e:
        print(f"OpenCV error: {e}")
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()
