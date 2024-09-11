# MoodPix: Real-time Emotion-driven Character Image Generator
MoodPix is a real-time emotion-driven character image generator using facial recognition and deep learning. It combines CNN-based emotion detection with GAN-based visual synthesis. The system transforms emotions into vibrant character visuals, leveraging OpenCV, TensorFlow, and GPU-accelerated generation techniques.

# Features

1. *Real-time Emotion Detection*: The core component utilizes *Convolutional Neural Networks (CNNs)* to perform accurate facial emotion classification in real-time. This is achieved through a highly optimized pipeline powered by *OpenCV* for facial detection and *TensorFlow/Keras* for emotion recognition.
   
2. *Prompt Generation*: The system dynamically generates relevant prompts based on detected emotions, leveraging advanced natural language algorithms. The *prompt.py* script translates emotional states into appropriate text-based prompts to guide the image generation process.

3. *Emotion-Based Character Image Generation*: By integrating *Generative Adversarial Networks (GANs)*, the tool produces high-quality character images reflecting the user’s mood. *GPU acceleration* and deep learning optimizations ensure that visuals are generated efficiently and at scale.

# Repository Structure

- **`realtimedetection.py`**: This script performs real-time emotion detection using facial recognition techniques. It uses *OpenCV* for face tracking and *TensorFlow/Keras* for emotion classification, ensuring highly responsive real-time processing.
  
- **`prompt.py`**: A utility for generating text-based prompts corresponding to detected emotions. This script helps bridge the gap between emotion detection and image generation by providing relevant prompts for mood-based content creation.

- **`generation.py`**: The primary script that integrates emotion detection with the generative model. It uses the output from the emotion detection pipeline and generates unique character images based on the identified mood. This script leverages *GANs* to create visually distinct characters, utilizing GPU-accelerated processing for faster generation times.

# Installation & Requirements

1. Clone the repository:
   ```
   git clone <repo-link>
   ```

2. Install the necessary dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Ensure that your environment is properly configured with **Python 3.x**, and necessary libraries such as *OpenCV*, *TensorFlow*, and *Keras*.

# Running the Application

Once all dependencies are installed, you can execute the following commands to run each module:

- **Real-time Emotion Detection**:
   ```
   python realtimedetection.py
   ```

- **Prompt Generation**:
   ```
   python prompt.py
   ```

- **Image Generation Based on Emotion**:
   ```
   python generation.py
   ```
