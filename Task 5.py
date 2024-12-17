import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Paths and Parameters
DATASET_DIR = 'C:\\Users\\kisho\\Downloads\\archive (2)\\food-101\\food-101\\images'  # Update with your dataset path
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 10

# Mapping Food Classes to Calorie Values (example, replace with real data)
food_calories = {
    'pizza': 266,
    'burger': 295,
    'salad': 150,
    'ice_cream': 207,
    'sushi': 200,
    # Add all food classes with calorie values
}

# Data Generators
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    subset='training',
    class_mode='categorical',
    shuffle=True  # This ensures the data is shuffled
)

val_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    subset='validation',
    class_mode='categorical',
    shuffle=True  # This ensures the data is shuffled
)

# Load Pre-trained Model (Transfer Learning)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
base_model.trainable = False  # Freeze base layers

# Add Custom Layers
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile Model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_steps=val_generator.samples // BATCH_SIZE
)

# Save the model
model.save('food_recognition_model.h5')

# Evaluate Model
val_loss, val_acc = model.evaluate(val_generator)
print(f"Validation Accuracy: {val_acc * 100:.2f}%")

# Plot Training History
plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Food Recognition and Calorie Estimation
def recognize_food_and_calories(image_path, model, food_calories, generator):
    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict Food Class
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions)
    predicted_class = generator.class_indices
    class_labels = {v: k for k, v in predicted_class.items()}
    food_name = class_labels[predicted_class_idx]
    
    # Get Calorie Value
    calories = food_calories.get(food_name, "Calorie info not available")
    
    return food_name, calories

# Test the Model
test_image_path = 'path_to_test_image.jpg'  # Update with your test image path
predicted_food, estimated_calories = recognize_food_and_calories(test_image_path, model, food_calories, train_generator)
print(f"Predicted Food: {predicted_food}")
print(f"Estimated Calories: {estimated_calories}")
