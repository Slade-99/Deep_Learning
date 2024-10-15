import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Concatenate
from tensorflow.keras.optimizers import Adam

# Set paths
train_dir = 'C:/Users/LENOVO/Downloads/Data/train'
valid_dir = 'C:/Users/LENOVO/Downloads/Data/test'

# Parameters
IMG_SIZE = 224  # SqueezeNet input size
BATCH_SIZE = 32
EPOCHS = 10  # Adjust based on your needs
LEARNING_RATE = 0.001
NUM_CLASSES = 3  # Update based on your number of classes

# Step 1: Data Preprocessing for Multiclass
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

# Step 2: Build the SqueezeNet Model
def Fire(x, squeeze=16, expand=64):
    """Fire module as described in the SqueezeNet paper."""
    # Squeeze
    x = Conv2D(squeeze, (1, 1), padding='same', activation='relu')(x)
    # Expand
    left = Conv2D(expand, (1, 1), padding='same', activation='relu')(x)
    right = Conv2D(expand, (3, 3), padding='same', activation='relu')(x)
    return Concatenate()([left, right])

def SqueezeNet(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    """Build SqueezeNet model."""
    inputs = Input(shape=input_shape)

    x = Conv2D(96, (7, 7), strides=(2, 2), padding='same', activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Fire modules
    x = Fire(x, 16, 64)
    x = Fire(x, 16, 64)
    x = Fire(x, 32, 128)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = Fire(x, 32, 128)
    x = Fire(x, 48, 192)
    x = Fire(x, 48, 192)
    x = Fire(x, 64, 256)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = Fire(x, 64, 256)

    # Global average pooling and classifier
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

# Build and compile the model
model = SqueezeNet(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES)

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

# Step 5: Save the Model
model.save('squeezenet_chest_xray_multiclass.h5')

# Step 6: Plot Training History
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

plot_history(history)

# Step 7: Evaluate Model with Confusion Matrix and Classification Report
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