import os
from PIL import Image
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Make dataset folder
#os.makedirs("Dataset", exist_ok=True)

# Extract images
#with zipfile.ZipFile("Dataset/archive (19).zip", 'r') as zip_ref:
    #zip_ref.extractall("project")

target_size = (150, 150)
dataset_folders = [
    r"C:\Users\smgop\Desktop\Task\FaceMask Detector\archive (19)\Dataset\mask_weared_incorrect",
    r"C:\Users\smgop\Desktop\Task\FaceMask Detector\archive (19)\Dataset\with_mask",
    r"C:\Users\smgop\Desktop\Task\FaceMask Detector\archive (19)\Dataset\without_mask"
]

# Resize function
def resize_image(folder_path):
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        try:
            img = Image.open(img_path)
            img = img.resize(target_size)
            img.save(img_path)
        except:
            print(f"Couldn't process image {img_path}")

for folder in dataset_folders:
    resize_image(folder)

# Data Generators
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    r"C:\Users\smgop\Desktop\Task\FaceMask Detector\archive (19)\Dataset",
    target_size=(150,150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    r"C:\Users\smgop\Desktop\Task\FaceMask Detector\archive (19)\Dataset",
    target_size=(150,150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# CNN Model
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_generator, validation_data=validation_generator, epochs=10)


model.save("C:/Users/smgop/Desktop/Task/FaceMask Detector/Face_Mask_cnn_model.h5")
