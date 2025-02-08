# For Data Processing
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from PIL import Image, ImageEnhance
import tensorflow as tf
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# For ML Models
from tensorflow.keras.layers import * 
from tensorflow.keras.models import * 
from tensorflow.keras.optimizers import * 
from tensorflow.keras.applications import VGG16 
from tensorflow.keras.preprocessing.image import load_img 

train_dir = '/home/azwad/Datasets/Benchmark_Dataset/Data/train/'
test_dir = '/home/azwad/Datasets/Benchmark_Dataset/Data/test/'

# Load training data
train_paths = []
train_labels = []
for label in os.listdir(train_dir):
    for image in os.listdir(train_dir + label):
        train_paths.append(train_dir + label + '/' + image)
        train_labels.append(label)

train_paths, train_labels = shuffle(train_paths, train_labels)

# Load testing data
test_paths = []
test_labels = []
for label in os.listdir(test_dir):
    for image in os.listdir(test_dir + label):
        test_paths.append(test_dir + label + '/' + image)
        test_labels.append(label)

test_paths, test_labels = shuffle(test_paths, test_labels)

# Data augmentation function
def augment_image(image):
    image = Image.fromarray(np.uint8(image))
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
    image = np.array(image) / 255.0
    return image

IMAGE_SIZE = 128

# Function to load and augment images
def open_images(paths):
    images = []
    for path in paths:
        image = load_img(path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        image = augment_image(image)
        images.append(image)
    return np.array(images)

unique_labels = os.listdir(train_dir)

def encode_label(labels):
    return np.array([unique_labels.index(x) for x in labels])

def decode_label(labels):
    return np.array([unique_labels[x] for x in labels])

def datagen(paths, labels, batch_size=12, epochs=1):
    for _ in range(epochs):
        for x in range(0, len(paths), batch_size):
            batch_paths = paths[x:x + batch_size]
            batch_images = open_images(batch_paths)
            batch_labels = labels[x:x + batch_size]
            batch_labels = encode_label(batch_labels)
            yield batch_images, batch_labels

# Parameters
NUM_CLIENTS = 10
NUM_ROUNDS = 7
batch_size = 32

# Prepare client data
clients = []
for i in range(NUM_CLIENTS):
    client_data = train_paths[i * (len(train_paths) // NUM_CLIENTS):(i + 1) * (len(train_paths) // NUM_CLIENTS)]
    client_labels = train_labels[i * (len(train_labels) // NUM_CLIENTS):(i + 1) * (len(train_labels) // NUM_CLIENTS)]
    clients.append((client_data, client_labels))

# Create the model
base_model = VGG16(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights='imagenet')

# Set layers to trainable
for layer in base_model.layers:
    layer.trainable = False
# Set the last few layers to trainable
base_model.layers[-2].trainable = True
base_model.layers[-3].trainable = True
base_model.layers[-4].trainable = True

model = Sequential()
model.add(Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
model.add(base_model)
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(len(unique_labels), activation='softmax'))

# Training process
for round_num in range(NUM_ROUNDS):

    # Select clients
    selected_client_indices = np.random.choice(len(clients), size=int(NUM_CLIENTS * 0.5), replace=False)
    selected_clients = [clients[i] for i in selected_client_indices]

    # Initialize G_i for Adagrad if not already initialized
    if 'G' not in locals():
        G = [np.zeros_like(weight) for weight in model.get_weights()]

    client_gradients = []

    for client in selected_clients:
        client_model = tf.keras.models.clone_model(model)
        client_model.set_weights(model.get_weights())
  
        # Compile the client model
        client_model.compile(optimizer=Adam(learning_rate=0.0001),
                             loss='sparse_categorical_crossentropy',
                             metrics=['sparse_categorical_accuracy'])

        steps_per_epoch = int(len(client[0]) / 20)
        
        # Train locally
        client_model.fit(datagen(client[0], client[1], batch_size=batch_size, epochs=1),
                         epochs=5, steps_per_epoch=steps_per_epoch)

        # Compute gradients only for trainable layers
        gradients = []
        trainable_layer_indices = [i for i, layer in enumerate(client_model.layers) if layer.trainable]
        
        for layer_index in trainable_layer_indices:
            grad = client_model.get_weights()[layer_index] - model.get_weights()[layer_index]
            gradients.append(grad)
        
        client_gradients.append(gradients)

    # Update G_i for Adagrad and aggregate gradients
    for layer_index in range(len(G)):
        if layer_index in trainable_layer_indices:
            for grad in client_gradients:
                G[layer_index] += grad[layer_index] ** 2
            
    # Update global model weights using aggregated gradients
    new_weights = model.get_weights()  # Start with the current weights

    for layer_index in trainable_layer_indices:
        # Calculate the adaptive learning rate for the trainable layer
        eta_i = 0.0001 / (np.sqrt(G[layer_index]) + 1e-8)  # Adjust learning rate calculation
        
        # Compute the average gradient for this layer
        avg_gradient = np.mean([grad[layer_index] for grad in client_gradients], axis=0)
        
        # Update weights for trainable layers only
        new_weights[layer_index] -= eta_i * avg_gradient

    # Set the new weights to the global model
    model.set_weights(new_weights)

# Evaluate the model
batch_size = 32
steps = int(len(test_paths) / batch_size)
y_pred = []
y_true = []

for x, y in tqdm(datagen(test_paths, test_labels, batch_size=batch_size, epochs=1), total=steps):
    pred = model.predict(x)
    pred = np.argmax(pred, axis=-1)
    y_pred.extend(decode_label(pred))
    y_true.extend(decode_label(y))

# Save the model
model.save('my_model.h5')

# Print classification report
print(classification_report(y_true, y_pred))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot the confusion matrix
font_size = 20
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=unique_labels, yticklabels=unique_labels,
            annot_kws={"fontsize": font_size}, cbar=False)
plt.xlabel("Predicted Label", fontsize=font_size)
plt.ylabel("True Label", fontsize=font_size)
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size, rotation=0)
plt.show()
