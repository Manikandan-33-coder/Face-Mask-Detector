from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

model = load_model("project/face_mask_Detection_cnn_model.h5")
upload_folder = 'static/uploads'
os.makedirs(upload_folder, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded!"
    file = request.files['file']
    filepath = os.path.join(upload_folder, file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    classes = ['Mask Worn Incorrectly', 'With Mask', 'Without Mask']

    result = f"Prediction: {classes[class_index]}"

    return render_template('index.html', prediction=result, image_file=file.filename)

if __name__ == "__main__":
    app.run(debug=True)
