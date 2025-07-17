from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load your trained model
model = load_model("project/face_mask_Detection_cnn_model.h5")

# Create upload folder
upload_folder = 'static/uploads'
os.makedirs(upload_folder, exist_ok=True)
app.config['UPLOAD_FOLDER'] = upload_folder

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded!"
    
    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Image processing
    img = image.load_img(filepath, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    classes = ['Mask Worn Incorrectly', 'With Mask', 'Without Mask']
    result = f"Prediction: {classes[class_index]}"

    return render_template('result.html', result=result, image_path=filepath)

# Run the app for local testing
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=10000)
