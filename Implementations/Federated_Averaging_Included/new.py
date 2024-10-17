### Imports   ###
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Lambda
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import os



##### Variables ######
train_dir = '/home/azwad/Datasets/Shortened/train'
valid_dir = '/home/azwad/Datasets/Shortened/test'
IMG_SIZE = 224  # MobileNetV2 requires images of size 224x224
BATCH_SIZE = 32
EPOCHS = 10  # Adjust based on your needs
LEARNING_RATE = 0.001
NUM_CLASSES = 3  # Update this based on the number of classes in your dataset





#### Preparing the Data  #####
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1./255)

# Load data from directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'  # Use categorical mode for multiclass classification
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'  # Use categorical mode for multiclass classification
)








#### Building the architecture  ######

def create_model():
    # Load MobileNetV2 without the top layers
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Modify the first convolutional layer to accept 1 channel
    first_layer = base_model.layers[0]
    weights = first_layer.get_weights()
    
    # Create a new input layer for grayscale
    input_layer = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)
    
    # Duplicate the grayscale channel to match the MobileNetV2 input shape
    x = tf.keras.layers.Lambda(lambda img: tf.image.grayscale_to_rgb(img))(x)
    
    # Use the base model with modified input
    x = base_model(x)
    
    # Add new layers on top
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    return Model(inputs=input_layer, outputs=predictions)

model = create_model()

model.trainable = True

# Step 3: Compile the Model
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',  # Use categorical crossentropy for multiclass classification
              metrics=['accuracy'])

# Step 4: Train the Model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // BATCH_SIZE,
    epochs=EPOCHS
)







# Step 6: Save the Model
model.save('mobilenet_chest_xray_multiclass.h5')

# Step 7: Plot Training History
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

plot_history(history_fine)

# Step 8: Evaluate Model with Confusion Matrix and Classification Report
valid_generator.reset()
preds = model.predict(valid_generator)
predicted_classes = np.argmax(preds, axis=1)  # Get the class with the highest probability

# Get true labels from the validation generator
true_classes = valid_generator.classes

# Get class labels
class_labels = list(valid_generator.class_indices.keys())

# Confusion Matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.show()

# Classification Report
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)