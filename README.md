
# 😷 Face Mask Detector

A deep learning-based application that detects whether a person is wearing a face mask properly, improperly, or not wearing one at all. This project uses a Convolutional Neural Network (CNN) model built with TensorFlow/Keras and OpenCV for real-time image processing.

---

## 📌 Features

- 📷 Detects faces in images.
- 😷 Classifies faces into three categories:
  - **Mask**
  - **Improper Mask**
  - **No Mask**
- 🖼️ Works on uploaded images.
- 🔍 Real-time detection support via webcam (optional extension).
- Simple and intuitive interface.

---

## 🗂️ Project Structure

```
Face-Mask-Detector/
├── Dataset/
│   ├── with_mask/
│   ├── without_mask/
│   └── mask_weared_incorrect/
├── model/
│   └── face_mask_detector.h5
├── app.py
├── requirements.txt
├── README.md
└── images/
    └── sample_test.jpg
```

---

## 🚀 Installation

1️⃣ **Clone the repository**

```bash
git clone https://github.com/Manikandan-33-coder/Face-Mask-Detector.git
cd Face-Mask-Detector
```

2️⃣ **Create a virtual environment (optional)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3️⃣ **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 🛠️ Usage

### 🖼️ Detect on Image

Run the detection app:

```bash
python app.py
```

- Upload an image through the interface.
- Get the prediction result on the image.

---

## 🧠 Model Details

- **Architecture:** Convolutional Neural Network (CNN)
- **Input Size:** 150x150 pixels
- **Training:** Trained on a dataset containing images of people with masks, without masks, and wearing masks improperly.
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Accuracy Achieved:** ~95% on validation data.

---

## 📦 Requirements

The required Python packages are listed in `requirements.txt`, including:

- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Flask (for web app)

Install them via:

```bash
pip install -r requirements.txt
```

---

## 📊 Sample Result

![Sample Output](images/sample_test.jpg)

---

## 📚 References

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Face Mask Detection Dataset (Kaggle)](https://www.kaggle.com/datasets)

---

## 📬 Contact

**Manikandan Murugan**  
[GitHub Profile](https://github.com/Manikandan-33-coder)  

---

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).

---

## ⭐ Support

If you like this project, give it a ⭐ on [GitHub](https://github.com/Manikandan-33-coder/Face-Mask-Detector) and share it with others!
