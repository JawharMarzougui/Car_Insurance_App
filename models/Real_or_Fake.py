import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# Function to load and preprocess images
def load_and_preprocess_images(folder_path, label):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            img_path = os.path.join(folder_path, filename)
            img = image.load_img(img_path, target_size=(224, 224))
            img = image.img_to_array(img)
            img = preprocess_input(img)
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

# Load and preprocess AI generated images
ai_images, ai_labels = load_and_preprocess_images('/content/Dataset/Ai_generated_images', 0)

# Load and preprocess real images
real_images, real_labels = load_and_preprocess_images('/content/Dataset/Damaged Vehicle Images/train', 1)

# Combine AI generated and real images
X = np.concatenate((ai_images, real_images), axis=0)
y = np.concatenate((ai_labels, real_labels), axis=0)

# Create VGG16 base model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a fully connected layer
x = Dense(256, activation='relu')(x)

# Add a binary classification layer
predictions = Dense(1, activation='sigmoid')(x)

# Build the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze VGG16 layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

# Save the trained model
model.save('../weights/image_classifier_model.h5')