
# Self-Driving Car Simulation Project

This project demonstrates a self-driving car simulation using a neural network model to predict steering angles from camera images. The system receives telemetry data from a simulator, processes the images, and sends real-time control commands (steering and throttle) to simulate autonomous driving.

## Project Overview

The project consists of two key components:

1. **Data Processing and Model Training** (Jupyter Notebook)
   - Preprocessing and augmenting training data.
   - Building and training the neural network model.
   - Saving the trained model for real-time deployment.

2. **Real-Time Simulation and Control** (Python Script)
   - Establishes communication with a driving simulator using Socket.IO.
   - Receives and preprocesses live images from the simulator.
   - Uses the pre-trained model to predict steering angles.
   - Dynamically adjusts throttle based on car speed and sends control commands to the simulator.

## Features
- Efficient real-time communication with the driving simulator.
- Preprocessing pipeline to enhance image data for model training and prediction.
- Neural network-based steering angle predictions.
- Dynamic throttle adjustment to maintain a steady driving speed.

## Project Structure
```
├── data/                   # Training data (images and telemetry logs)
├── notebooks/              # Jupyter Notebooks for data processing and model training
│   └── SelfDrivingCar.ipynb
├── model/                  # Pre-trained model
│   └── model.h5
├── drive.py                # Main script for real-time car control
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Prerequisites
To set up this project, you'll need the following software and libraries:
- Python 3.7+
- Jupyter Notebook
- TensorFlow and Keras
- Flask
- python-socketio
- OpenCV
- Pillow (PIL)
- NumPy
- Eventlet

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Getting Started

### 1. Data Preparation and Training (Jupyter Notebook)
1. **Open the Jupyter Notebook:**
   - Navigate to the `notebooks/` directory and open `SelfDrivingCar.ipynb`.
   - This notebook contains code to load, preprocess, and augment driving data.
   
2. **Train the Model:**
   - Follow the steps to build and train a convolutional neural network.
   - Save the trained model as `model.h5` in the `model/` directory.
   
3. **Data Preprocessing:**
   - **Image Cropping and Resizing:** Remove irrelevant parts of images (e.g., sky and car hood) and resize them.
   - **Color Conversion:** Convert RGB images to YUV for better feature extraction.
   - **Normalization:** Normalize pixel values to a range between 0 and 1 for faster convergence.

### 2. Running the Real-Time Simulation (Python Script)
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/self-driving-car-sim.git
   cd self-driving-car-sim
   ```

2. **Ensure the trained model is available at `model/model.h5`.**

3. **Start the Simulation Server:**
   ```bash
   python drive.py
   ```
   - The server will connect to the driving simulator and control the car based on real-time predictions.

### How it Works:
- **Establish Connection:** The script uses Socket.IO to connect to the simulator.
- **Receive Telemetry Data:** It processes telemetry data and images from the car’s front camera.
- **Predict Steering Angle:** The pre-trained neural network model predicts the steering angle based on the preprocessed image.
- **Adjust Throttle:** A dynamic throttle adjustment helps to keep the car within a speed limit.

## Code Explanation

### From `drive.py`
- **`img_preprocess(img)`:** 
   - Crops the image to remove unnecessary parts, resizes it to 200x66 pixels, converts it to YUV color space, applies Gaussian Blur, and normalizes the values.
- **`telemetry(sid, data)`:** 
   - Handles incoming data, preprocesses the image, uses the model to predict the steering angle, and adjusts the throttle to control the car.
- **`send_control(steering_angle, throttle)`:** 
   - Sends steering and throttle commands back to the simulator for real-time driving.

### From `SelfDrivingCar.ipynb`
- **Data Visualization and Augmentation:**
   - Visualizes driving data to identify patterns.
   - Applies data augmentation (flipping, brightness adjustment) to enhance training data.
- **Model Training:**
   - Builds a Convolutional Neural Network (CNN) for predicting steering angles.
   - Compiles and trains the model using Mean Squared Error (MSE) as the loss function.
   - Saves the trained model for use during real-time simulation.

## Example Output
```plaintext
Connected: abc123
Steering Angle: 0.032, Throttle: 0.85, Speed: 9.2
Steering Angle: -0.045, Throttle: 0.78, Speed: 8.5
...
```

## Troubleshooting
1. **Server Connection Issues:**
   - Ensure the simulator and the server are on the same network.
   - Check that the correct IP address and port are configured.
   
2. **Model Loading Issues:**
   - Verify that `model.h5` exists in the `model/` directory.
   - Make sure TensorFlow/Keras versions are compatible with the model.

3. **Performance Issues:**
   - Try optimizing the neural network architecture in the Jupyter Notebook.
   - Use a higher quality dataset for better model performance.

## To-Do List / Future Enhancements
- Improve the data preprocessing pipeline to handle more diverse lighting conditions.
- Optimize the neural network architecture for better accuracy.
- Add more telemetry parameters (e.g., braking) for finer control.
- Implement lane detection for improved path planning.

## Contributing
Contributions are welcome! Please follow the steps below:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and open a pull request.
