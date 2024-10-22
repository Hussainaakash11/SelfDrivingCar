import os
import socketio
import eventlet
import numpy as np
from flask import Flask
from keras.models import load_model
from keras.saving import register_keras_serializable
from keras.losses import MeanSquaredError
import base64
from io import BytesIO
from PIL import Image
import cv2

# Disable oneDNN custom operations (optional)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Register the custom mse function
@register_keras_serializable()
def mse(y_true, y_pred):
    return MeanSquaredError()(y_true, y_pred)

# Initialize Socket.IO server and Flask application
sio = socketio.Server()
app = Flask(__name__)

# Set a speed limit
speed_limit = 30

def img_preprocess(img):
    """Preprocess the image for model input."""
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    return img

@sio.on('connect')
def connect(sid, environ):
    print('Connected:', sid)
    send_control(0, 0)

def send_control(steering_angle, throttle):
    """Send steering and throttle control to the simulator."""
    sio.emit('steer', data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    })

@sio.on('telemetry')
def telemetry(sid, data):
    try:
        # Get current speed and image data
        speed = float(data['speed'])
        image = Image.open(BytesIO(base64.b64decode(data['image'])))
        image = np.asarray(image)
        
        # Preprocess the image
        image = img_preprocess(image)
        image = np.array([image])
        
        # Predict steering angle using the model
        steering_angle = float(model.predict(image))
        throttle = 1.0 - speed / speed_limit
        
        print(f'Steering Angle: {steering_angle}, Throttle: {throttle}, Speed: {speed}')
        
        # Send control commands to the simulator
        send_control(steering_angle, throttle)
        
    except Exception as e:
        print(f'Error during telemetry processing: {e}')
        send_control(0, 0)

if __name__ == '__main__':
    try:
        # Load the trained model with custom objects
        model = load_model('model.h5', custom_objects={'mse': mse})
        
        # Wrap the application with Socket.IO middleware
        app = socketio.Middleware(sio, app)
        
        # Start the Eventlet web server
        eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
    except Exception as e:
        print(f'Error starting the server: {e}')
