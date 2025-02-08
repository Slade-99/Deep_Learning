### Imports ###
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import random
import seaborn as sns
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis
import os
from tensorflow.keras.preprocessing.image import load_img
import cv2
from sklearn.utils import shuffle
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Concatenate





##### Variables ######
train_dir = '/home/azwad/Downloads/Dataset_Finalized/train/'
test_dir = '/home/azwad/Downloads/Dataset_Finalized/test/'
log_file_name = "Federated Learning Training Log for SqueezeNet\n"
log_file_path = "training_logs_SqueezeNet.txt"
train_paths = []
train_labels = []
test_paths = []
test_labels = []
IMG_SIZE = 224
LEARNING_RATE = 0.0001
NUM_CLIENTS = 2
NUM_ROUNDS = 1
NUM_CLASSES = 3
batch_size = 32
steps = int(len(train_paths)/batch_size)
epochs = 1




#### Dataset Preparation ######
def eval_datagen(paths, labels, batch_size=12):
    while True:
        for x in range(0, len(paths), batch_size):
            batch_paths = paths[x:x + batch_size]
            batch_images = open_images(batch_paths)
            batch_labels = labels[x:x + batch_size]
            batch_labels = encode_label(batch_labels)
            yield batch_images, batch_labels
for label in os.listdir(train_dir):
    for image in os.listdir(train_dir+label):
        train_paths.append(train_dir+label+'/'+image)
        train_labels.append(label)
train_paths, train_labels = shuffle(train_paths, train_labels)
for label in os.listdir(test_dir):
    for image in os.listdir(test_dir+label):
        test_paths.append(test_dir+label+'/'+image)
        test_labels.append(label)
test_paths, test_labels = shuffle(test_paths, test_labels)
def augment_image(image):
    # Convert to NumPy array if it's a PIL Image
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray_image)
    
    # Apply median blur
    blurred_image = cv2.medianBlur(clahe_image, ksize=5)  # ksize must be odd
    
    # Normalize the image to [0, 1]
    normalized_image = blurred_image / 255.0
    normalized_image = np.stack((normalized_image,) * 3, axis=-1)
    return normalized_image
def open_images(paths):
    '''
    Given a list of paths to images, this function returns the images as arrays (after augmenting them)
    '''
    images = []
    for path in paths:
        image = load_img(path, target_size=(IMG_SIZE,IMG_SIZE))
        image = augment_image(image)
        images.append(image)
    return np.array(images)

unique_labels = os.listdir(train_dir)

def encode_label(labels):
    encoded = []
    for x in labels:
        encoded.append(unique_labels.index(x))
    return np.array(encoded)

def decode_label(labels):
    decoded = []
    for x in labels:
        decoded.append(unique_labels[x])
    return np.array(decoded)

def datagen(paths, labels, batch_size=12, epochs=1):
    for _ in range(epochs):
        for x in range(0, len(paths), batch_size):
            batch_paths = paths[x:x+batch_size]
            batch_images = open_images(batch_paths)
            batch_labels = labels[x:x+batch_size]
            batch_labels = encode_label(batch_labels)
            yield batch_images, batch_labels
            
eval_generator = eval_datagen(test_paths, test_labels, batch_size=batch_size)








### Federated Learning Setup #####
clients = []
for i in range(NUM_CLIENTS):
    client_data = train_paths[i * (len(train_paths) // NUM_CLIENTS):(i + 1) * (len(train_paths) // NUM_CLIENTS)]
    client_labels = train_labels[i * (len(train_labels) // NUM_CLIENTS):(i + 1) * (len(train_labels) // NUM_CLIENTS)]
    clients.append((client_data, client_labels))












####    Architecture #####
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

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',  # Use categorical crossentropy for multiclass classification
              metrics=['accuracy'])













#### Federated Learning Training Loop #####
for round_num in range(NUM_ROUNDS):

    # Select clients
    selected_client_indices = np.random.choice(len(clients), size=int(NUM_CLIENTS * 0.5), replace=False)
    selected_clients = [clients[i] for i in selected_client_indices]

    client_losses = []
    client_accuracies = []
    # Transmit the global model to the selected clients
    for client in selected_clients:
        client_model = tf.keras.models.clone_model(model)
        client_model.set_weights(model.get_weights())
  
        # Compile the client model
        client_model.compile(optimizer=Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'])

        steps_per_epoch = int(len(client[0]) / 20)
        
        # Train locally
        history = client_model.fit(datagen(client[0],client[1], batch_size=batch_size, epochs=epochs),
                         epochs=5, steps_per_epoch=steps_per_epoch)

        client_losses.append(history.history['loss'][-1])
        client_accuracies.append(history.history['sparse_categorical_accuracy'][-1])
        
        steps_per_epoch = int(len(client[0]) / 20)

        # Aggregate the model
        new_weights = []
        for layer_index in range(len(model.get_weights())):
            new_layer_weights = np.mean([client_model.get_weights()[layer_index], model.get_weights()[layer_index]], axis=0)
            new_weights.append(new_layer_weights)
        model.set_weights(new_weights)
        
    loss = np.mean(client_losses)
    accuracy = np.mean(client_accuracies)
    with open(log_file_path, 'a') as f:
        f.write(f"Round {round_num + 1}, Client Loss: {loss:.4f}, Client Accuracy: {accuracy:.4f}\n")

    model_save_path = 'squeezenet_chest_xray_multiclass.h5'
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")










###### Evaluation ########
test_images = open_images(test_paths)
test_labels_encoded = encode_label(test_labels)

# Perform predictions
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)

report = classification_report(test_labels_encoded, predicted_classes, 
                               target_names=unique_labels, 
                               output_dict=True, 
                               zero_division=0) 
# Extract metrics
accuracy = report['accuracy']
precision = report['weighted avg']['precision']
recall = report['weighted avg']['recall']
f1_score = report['weighted avg']['f1-score']

with open(log_file_path, 'a') as f:
    f.write(f"Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}\n")

plt.figure(figsize=(10, 8))
cm = confusion_matrix(test_labels_encoded, predicted_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')  # Save the confusion matrix as an image
plt.close()

def count_flops(model):
    flops = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            # FLOPs for Conv2D: (filter_height * filter_width * input_channels + 1) * output_height * output_width * output_channels
            flops += layer.kernel_size[0] * layer.kernel_size[1] * layer.input_shape[-1] * layer.output_shape[1] * layer.output_shape[2] * layer.filters
        elif isinstance(layer, tf.keras.layers.Dense):
            # FLOPs for Dense: (input_units + 1) * output_units
            flops += (layer.input_shape[-1] + 1) * layer.units
        # Add more layer types as needed...
    return flops

flops = count_flops(model)
with open(log_file_path, 'a') as f:
    f.write(f"FLOPs: {flops}\n")
start_time = time.time()
model.predict(test_images)  # Run inference on the test set
latency = time.time() - start_time
with open(log_file_path, 'a') as f:
    f.write(f"Latency: {latency:.4f} seconds\n")






