# Traffic-sign-recognition

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from collections import Counter
from sklearn.metrics import (confusion_matrix, classification_report, 
                           roc_curve, auc, precision_recall_curve)
from sklearn.preprocessing import label_binarize
from itertools import cycle
from PIL import Image
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, 
                                   Dropout, BatchNormalization, Activation,
                                   Add, GlobalAveragePooling2D, Input,
                                   DepthwiseConv2D, AveragePooling2D)
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.utils import to_categorical

# Define paths
data_dir = "C:/Users/LENOVO/Downloads/GRSRB/"
train_path = os.path.join(data_dir, "Train")
test_path = os.path.join(data_dir, "Test")
test_csv_path = os.path.join(test_path, "GT-final_test.csv")

# Parameters
img_size = (28, 28)  # Resize images to 28x28
num_classes = 43      # Number of classes in GTSRB
batch_size = 32       # Batch size for training
epochs = 10           # Number of epochs

# Function to load images and labels from a folder
def load_images_from_folder(folder, img_size):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_path = os.path.join(folder, label)
        if os.path.isdir(label_path):
            for image_name in os.listdir(label_path):
                if image_name.endswith('.png'):  # Only process .png files
                    image_path = os.path.join(label_path, image_name)
                    try:
                        image = Image.open(image_path).resize(img_size)
                        image = np.array(image) / 255.0  # Normalize to [0, 1]
                        images.append(image)
                        labels.append(int(label))
                    except Exception as e:
                        print(f"Warning: Unable to load image {image_path}. Error: {e}")
    return np.array(images), np.array(labels)

# Load training data
X_train, y_train = load_images_from_folder(train_path, img_size)

# Load test data
test_data = pd.read_csv(test_csv_path, sep=';')
X_test = []
y_test = test_data['ClassId'].values
for image_name in test_data['Filename']:
    # Replace .ppm with .png in the filename
    image_name = image_name.replace('.ppm', '.png')
    image_path = os.path.join(test_path, image_name)
    try:
        image = Image.open(image_path).resize(img_size)
        image = np.array(image) / 255.0  # Normalize to [0, 1]
        X_test.append(image)
    except Exception as e:
        print(f"Warning: Unable to load image {image_path}. Error: {e}")
        
X_test = np.array(X_test)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Calculate class weights
class_counts = Counter(y_train)  # Count occurrences of each class
total_samples = sum(class_counts.values())
class_weights = {cls: total_samples / (num_classes * count) for cls, count in class_counts.items()}

# GTSRB Class Labels
class_names = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
    9: 'No passing', 10: 'No passing veh over 3.5 tons', 11: 'Right-of-way at intersection',
    12: 'Priority road', 13: 'Yield', 14: 'Stop', 15: 'No vehicles', 16: 'Veh > 3.5 tons prohibited',
    17: 'No entry', 18: 'General caution', 19: 'Dangerous curve left', 20: 'Dangerous curve right',
    21: 'Double curve', 22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows on the right',
    25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing',
    32: 'End speed + passing limits', 33: 'Turn right ahead', 34: 'Turn left ahead',
    35: 'Ahead only', 36: 'Go straight or right', 37: 'Go straight or left', 38: 'Keep right',
    39: 'Keep left', 40: 'Roundabout mandatory', 41: 'End of no passing', 42: 'End no passing veh > 3.5 tons'
}

# Display class distribution
# Color-blind friendly palette (Okabe-Ito)
cb_palette = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', 
               '#0072B2', '#D55E00', '#CC79A7', '#999999']

plt.figure(figsize=(15, 8))
bars = plt.bar(class_names.values(), class_counts.values(), color=cb_palette)
plt.title('Class Distribution in Training Set')
plt.xlabel('Traffic Sign Classes')
plt.ylabel('Number of Images')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Display sample images from each class
plt.figure(figsize=(15, 15))
for i in range(num_classes):
    plt.subplot(7, 7, i+1)
    class_indices = np.where(y_train == i)[0]
    if len(class_indices) > 0:
        sample_idx = class_indices[0]
        plt.imshow(X_train[sample_idx])
        plt.title(f'Class {i}')
        plt.axis('off')
plt.tight_layout()
plt.suptitle('Sample Images from Each Class', y=1.02)
plt.show()

# Custom Model
def build_basic_cnn():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=RMSprop(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 2. Mini-ResNet Model
def mini_resnet_block(input_tensor, filters):
    x = Conv2D(filters, (3, 3), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    
    shortcut = Conv2D(filters, (1, 1), padding='same')(input_tensor)
    shortcut = BatchNormalization()(shortcut)
    
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def build_mini_resnet():
    inputs = Input(shape=(28, 28, 3))
    
    # Initial conv layer
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Residual blocks
    x = mini_resnet_block(x, 32)
    x = MaxPooling2D((2, 2))(x)
    
    x = mini_resnet_block(x, 64)
    x = MaxPooling2D((2, 2))(x)
    
    x = mini_resnet_block(x, 128)
    x = MaxPooling2D((2, 2))(x)
    
    # Final layers
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=RMSprop(learning_rate=0.0001),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    return model

# 3. Mini-MobileNet Model
def mini_mobilenet_block(x, filters, strides=1):
    # Depthwise convolution
    x = DepthwiseConv2D((3, 3), padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Pointwise convolution
    x = Conv2D(filters, (1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def build_mini_mobilenet():
    inputs = Input(shape=(28, 28, 3))
    
    # Initial conv layer
    x = Conv2D(32, (3, 3), padding='same', strides=(2, 2))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # MobileNet blocks
    x = mini_mobilenet_block(x, 64)
    x = MaxPooling2D((2, 2))(x)
    
    x = mini_mobilenet_block(x, 128)
    x = MaxPooling2D((2, 2))(x)
    
    x = mini_mobilenet_block(x, 256)
    x = MaxPooling2D((2, 2))(x)
    
    # Final layers
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=RMSprop(learning_rate=0.0001),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    return model
    
# 4. LeNet Model
def build_lenet():
    model = Sequential([
        Conv2D(6, (5, 5), activation='relu', input_shape=(28, 28, 3)),
        AveragePooling2D((2, 2)),
        Conv2D(16, (5, 5), activation='relu'),
        AveragePooling2D((2, 2)),
        Flatten(),
        Dense(120, activation='relu'),
        Dense(84, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    return model

# Create models
models = {
    'BasicCNN': build_basic_cnn(),
    'MiniResNet': build_mini_resnet(),
    'MiniMobileNet': build_mini_mobilenet(),
    'LeNet': build_lenet()
}

# Train and evaluate models
histories = {}
results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    history = model.fit(
        X_train, y_train_onehot,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test_onehot),
        class_weight=class_weights,
        verbose=1
    )
    histories[name] = history
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test_onehot, verbose=0)
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    

    
    # Save model
    model.save(f"{name}_traffic_sign_model.h5")


# Define model names (based on what you used while saving)
model_names = ['BasicCNN', 'MiniResNet', 'MiniMobileNet', 'LeNet']

# Dictionary to store classification reports
classification_reports = {}

# Loop through each saved model
for name in model_names:
    print(f"\nLoading {name} model and generating classification report...")
    
    # Load model
    model = load_model(f"{name}_traffic_sign_model.h5")
    
    # Predict
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test_onehot, axis=1)

    # Generate classification report and save it
    report = classification_report(y_true, y_pred_classes, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    classification_reports[name] = report_df

# Display all reports
for name, df in classification_reports.items():
    print(f"\nClassification Report for {name}:\n")
    display(df.round(4))  # Use print(df.to_string())
# After training each model, add these lines to print accuracy:
for name, model in models.items():
    print(f"\nEvaluating {name}...")
    test_loss, test_acc = model.evaluate(X_test, y_test_onehot, verbose=0)
    print(f"{name} Test Accuracy: {test_acc*100:.2f}%")
    
    # Print training accuracy from the last epoch
    last_train_acc = histories[name].history['accuracy'][-1]
    print(f"{name} Final Training Accuracy: {last_train_acc*100:.2f}%")
    
    # Print validation accuracy from the last epoch
    last_val_acc = histories[name].history['val_accuracy'][-1]
    print(f"{name} Final Validation Accuracy: {last_val_acc*100:.2f}%")

 # Color-blind friendly palette (Okabe-Ito)
cb_palette = ['#E69F00', '#56B4E9', '#009E73', '#0072B2', 
              '#D55E00', '#CC79A7', '#F0E442', '#000000']

# Line styles and markers for extra distinction
line_styles = ['-', '--', '-.', ':']
markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']

plt.figure(figsize=(15, 10))
for i, (name, history) in enumerate(histories.items()):
    color = cb_palette[i % len(cb_palette)]
    linestyle = line_styles[i % len(line_styles)]
    marker = markers[i % len(markers)]
    
    # Training accuracy (dashed + marker)
    plt.plot(history.history['accuracy'], 
             linestyle='--', 
             marker=marker,
             color=color,
             label=f'{name} Training')
    
    # Validation accuracy (solid + same marker)
    plt.plot(history.history['val_accuracy'], 
             linestyle='-', 
             marker=marker,
             color=color,
             label=f'{name} Validation')

plt.title('Model Training & Validation Accuracy', fontsize=14)
plt.ylabel('Accuracy', fontsize=12)
plt.xlabel('Epoch', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside
plt.grid(alpha=0.3)  # Lighter grid lines
plt.ylim(0, 1)  # Fix y-axis for accuracy metrics
plt.tight_layout()
plt.show()

# Color-blind friendly palette (Okabe-Ito) + styles
cb_palette = ['#E69F00', '#56B4E9', '#009E73', '#0072B2', 
              '#D55E00', '#CC79A7', '#F0E442', '#000000']

line_styles = ['-', '--', '-.', ':']  # Solid, dashed, etc.
markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']  # Shapes for extra distinction

plt.figure(figsize=(15, 10))
for i, (name, history) in enumerate(histories.items()):
    color = cb_palette[i % len(cb_palette)]
    linestyle = line_styles[i % len(line_styles)]
    marker = markers[i % len(markers)]
    
    # Training loss (dashed + marker)
    plt.plot(history.history['loss'], 
             linestyle='--', 
             marker=marker,
             color=color,
             label=f'{name} Training')
    
    # Validation loss (solid + same marker)
    plt.plot(history.history['val_loss'], 
             linestyle='-', 
             marker=marker,
             color=color,
             label=f'{name} Validation')

plt.title('Model Training & Validation Loss', fontsize=14)
plt.ylabel('Loss', fontsize=12)
plt.xlabel('Epoch', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend outside
plt.grid(alpha=0.3)  # Light grid
plt.ylim(bottom=0)  # Start y-axis at 0 for loss
plt.tight_layout()
plt.show()

# GTSRB Class Labels
class_names = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
    9: 'No passing', 10: 'No passing veh over 3.5 tons', 11: 'Right-of-way at intersection',
    12: 'Priority road', 13: 'Yield', 14: 'Stop', 15: 'No vehicles', 16: 'Veh > 3.5 tons prohibited',
    17: 'No entry', 18: 'General caution', 19: 'Dangerous curve left', 20: 'Dangerous curve right',
    21: 'Double curve', 22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows on the right',
    25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing',
    32: 'End speed + passing limits', 33: 'Turn right ahead', 34: 'Turn left ahead',
    35: 'Ahead only', 36: 'Go straight or right', 37: 'Go straight or left', 38: 'Keep right',
    39: 'Keep left', 40: 'Roundabout mandatory', 41: 'End of no passing', 42: 'End no passing veh > 3.5 tons'
}

# Example prediction on a single image
image_path = "C:/Users/LENOVO/Downloads/traffic5.png"
img = Image.open(image_path).resize((28, 28))
img_array = np.array(img) / 255.0  # Normalize

# Handle grayscale and add batch dimension
if len(img_array.shape) == 2:
    img_array = np.stack((img_array,)*3, axis=-1)
img_array = np.expand_dims(img_array, axis=0)

# Make predictions with all models
predictions = {}
for name, model in models.items():
    predictions[name] = model.predict(img_array)

# Display predictions
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Test Image")
plt.axis('off')

plt.subplot(1, 2, 2)
for name, pred in predictions.items():
    pred_class = np.argmax(pred)
    confidence = np.max(pred)
    plt.barh([name], [confidence*100], alpha=0.6)
    plt.text(confidence*100, list(predictions.keys()).index(name), 
             f"{class_names[pred_class]} ({confidence:.2%})", 
             ha='left', va='center')

plt.title('Model Predictions Comparison')
plt.xlabel('Confidence (%)')
plt.xlim(0, 100)
plt.tight_layout()
plt.show()

# =============================================
# Training Time Comparison (Balanced Style)
# =============================================
training_times = {
    "Basic CNN": 274,  # Example (replace with actual training times)
    "Mini ResNet": 1955,
    "MinimobileNet": 312,
    "LeNet":119
}

plt.figure(figsize=(8, 5))
sns.barplot(x=list(training_times.keys()), y=list(training_times.values()))
plt.title('Training Time (Seconds)')
plt.ylabel('Time (s)')
plt.xticks(rotation=45)
plt.show()

# =============================================
# Model Size Comparison (Balanced Style)
# =============================================
model_sizes = {}
for name, model in models.items():
    model.save(f'temp_{name}.h5')
    model_sizes[name] = os.path.getsize(f'temp_{name}.h5') / (1024 * 1024)  # Convert to MB

plt.figure(figsize=(10, 6))
sns.barplot(x=list(model_sizes.keys()), 
            y=list(model_sizes.values()),
            palette='plasma',
            edgecolor='black',
            linewidth=1)

plt.title('Model Size (MB)')
plt.ylabel('Size (MB)')
plt.xticks(rotation=45)
plt.show()

# =============================================
# Inference Time Comparison (Balanced Style)
# =============================================
inference_times = {}
for name, model in models.items():
    start = time()
    model.predict(X_test[:1])  # Predict a single image
    inference_times[name] = (time() - start) * 1000  # Convert to milliseconds

plt.figure(figsize=(8, 5))
sns.barplot(x=list(inference_times.keys()), 
            y=list(inference_times.values()), 
            palette='magma',
            edgecolor='black',
            linewidth=1)
plt.title('Inference Time (ms per Image)')
plt.ylabel('Time (ms)')
plt.xticks(rotation=45)
plt.show()
