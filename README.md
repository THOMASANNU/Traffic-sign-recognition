# Traffic-sign-recognition

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model,save_model, load_model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, AveragePooling2D, 
                                    Dense, Flatten, Dropout, BatchNormalization,
                                    Activation, Input, Add, GlobalAveragePooling2D,
                                    DepthwiseConv2D)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# Define paths
data_dir = "C:/Users/LENOVO/Downloads/GRSRB/"
train_path = os.path.join(data_dir, "Train")
test_path = os.path.join(data_dir, "Test")
test_csv_path = os.path.join(test_path, "GT-final_test.csv")

# Parameters
img_size = (28, 28)  # Resize images to 28x28
num_classes = 43      # Number of classes in GTSRB
batch_size = 32       # Batch size for training
epochs = 20            # Number of epochs

# Enhanced data loading with minority class augmentation
def load_images_with_augmentation(folder, img_size, augment_minority=True):
    images = []
    labels = []
    class_counts = Counter()
    
    # First pass: count class occurrences
    for label in os.listdir(folder):
        label_path = os.path.join(folder, label)
        if os.path.isdir(label_path):
            class_counts[int(label)] = len([name for name in os.listdir(label_path) 
                                         if name.endswith('.png')])
    
    # Determine median class count for augmentation strategy
    median_count = np.median(list(class_counts.values()))
    
    # Second pass: load images with augmentation strategy
    for label in os.listdir(folder):
        label_path = os.path.join(folder, label)
        if os.path.isdir(label_path):
            label_int = int(label)
            is_minority = class_counts[label_int] < median_count
            
            for image_name in os.listdir(label_path):
                if image_name.endswith('.png'):
                    image_path = os.path.join(label_path, image_name)
                    try:
                        image = Image.open(image_path).resize(img_size)
                        image = np.array(image) / 255.0
                        
                        if len(image.shape) == 2:
                            image = np.stack((image,)*3, axis=-1)
                        
                        images.append(image)
                        labels.append(label_int)
                        
                        # Add augmented versions for minority classes
                        if augment_minority and is_minority:
                            # Horizontal flip
                            flipped = np.fliplr(image)
                            images.append(flipped)
                            labels.append(label_int)
                            
                            # Slight rotation (10 degrees)
                            rotated = np.array(Image.fromarray((image*255).astype(np.uint8))
                                              .rotate(10, resample=Image.BILINEAR)) / 255.0
                            images.append(rotated)
                            labels.append(label_int)
                            
                            # Slight zoom
                            zoomed = image[2:26, 2:26]  # Crop center
                            zoomed = np.array(Image.fromarray((zoomed*255).astype(np.uint8))
                                              .resize((28, 28))) / 255.0
                            images.append(zoomed)
                            labels.append(label_int)
                            
                    except Exception as e:
                        print(f"Warning: Unable to load image {image_path}. Error: {e}")
    
    return np.array(images), np.array(labels)

# Load training data with minority class augmentation
print("Loading training data...")
X_train, y_train = load_images_with_augmentation(train_path, img_size, augment_minority=True)

# Load test data
print("Loading test data...")
test_data = pd.read_csv(test_csv_path, sep=';')
X_test = []
y_test = test_data['ClassId'].values
for image_name in test_data['Filename']:
    image_name = image_name.replace('.ppm', '.png')
    image_path = os.path.join(test_path, image_name)
    try:
        image = Image.open(image_path).resize(img_size)
        image = np.array(image) / 255.0
        if len(image.shape) == 2:
            image = np.stack((image,)*3, axis=-1)
        X_test.append(image)
    except Exception as e:
        print(f"Warning: Unable to load image {image_path}. Error: {e}")
        
X_test = np.array(X_test)

# Convert labels to one-hot encoding
y_train_onehot = to_categorical(y_train, num_classes)
y_test_onehot = to_categorical(y_test, num_classes)

# Calculate class weights

class_weights = class_weight.compute_class_weight('balanced',
                                                classes=np.unique(y_train),
                                                y=y_train)
class_weights = {i: class_weights[i] for i in range(len(class_weights))}

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

class_counts = Counter(y_train)

plt.figure(figsize=(15, 8))
bars = plt.bar(class_names.values(), class_counts.values(), color=cb_palette)
plt.title('Class Distribution in Training Set')
plt.xlabel('Traffic Sign Classes')
plt.ylabel('Number of Images')
plt.xticks(rotation=90)

# Add numbers on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             '%d' % int(height),
             ha='center', va='bottom')

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

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Custom Model
def build_custom_cnn():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),
        
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    return model


# 2. Mini-ResNet Model
def mini_resnet_block(input_tensor, filters, strides=1):
    x = Conv2D(filters, (3, 3), padding='same', strides=strides)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    
    if strides != 1 or input_tensor.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), padding='same', strides=strides)(input_tensor)
        shortcut = BatchNormalization()(shortcut)
    else:
        shortcut = input_tensor
    
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
    model.compile(optimizer=Adam(learning_rate=0.001),
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
    x = Conv2D(32, (3, 3), padding='same', strides=(1, 1))(inputs)
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
    model.compile(optimizer=RMSprop(learning_rate=0.001),
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

# Data augmentation pipeline
def create_augmentation_pipeline():
    minority_augmenter = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    majority_augmenter = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.1,
        fill_mode='nearest'
    )
    
    return minority_augmenter, majority_augmenter

# Modified training function to handle class weights with generator
def train_with_augmented_data(model, model_name, X_train, y_train, X_test, y_test, epochs, batch_size):
    minority_augmenter, majority_augmenter = create_augmentation_pipeline()
    
    # Calculate median class count
    class_counts = Counter(y_train)
    median_count = np.median(list(class_counts.values()))
    
    # Split data into minority and majority
    minority_indices = [i for i, label in enumerate(y_train) 
                       if class_counts[label] < median_count]
    majority_indices = [i for i, label in enumerate(y_train)
                       if class_counts[label] >= median_count]
    
    X_minority = X_train[minority_indices]
    y_minority = y_train[minority_indices]
    X_majority = X_train[majority_indices]
    y_majority = y_train[majority_indices]
    
    # Apply class weights directly to the samples
    sample_weights = np.ones(len(y_train))
    
    
    # Create generators with sample weights
    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Combine all data back since we can't use separate generators with sample weights
    train_generator = train_datagen.flow(
        X_train,
        to_categorical(y_train, num_classes),
        sample_weight=sample_weights,
        batch_size=batch_size,
        shuffle=True
    )
    # Calculate proper steps per epoch
    steps_per_epoch = len(X_train) // batch_size
    if len(X_train) % batch_size != 0:
        steps_per_epoch += 1  # Add one more step for remaining samples
    
    print(f"\nTraining {model_name}...")
    history = model.fit(
        train_generator,
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test_onehot),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    return history
    
# Train and evaluate models
histories = {}
results = []

# Define save directory 
save_dir = "C:/Users/LENOVO/Downloads/GRSRB/saved_models/"
os.makedirs(save_dir, exist_ok=True)

for name, model in models.items():
    history = train_with_augmented_data(
        model, name, X_train, y_train, X_test, y_test_onehot,
        epochs=epochs, batch_size=batch_size
    )
    histories[name] = history
    
    # Evaluation
    test_loss, test_acc = model.evaluate(X_test, y_test_onehot, verbose=0)
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    report = classification_report(np.argmax(y_test_onehot, axis=1), y_pred_classes, output_dict=True)
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1 = report['weighted avg']['f1-score']
    
    results.append({
        'Model': name,
        'Accuracy': test_acc,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    
    # Save model in single .keras format
    model_path = os.path.join(save_dir, f"{name}_traffic_sign_model.keras")
    save_model(model, model_path)
    print(f"{name} saved successfully at: {model_path}")


# Save evaluation results (optional)
results_df = pd.DataFrame(results).set_index('Model')

# Convert results to DataFrame and display
results_df = pd.DataFrame(results).set_index('Model')
print("\nModel Performance Summary:")
print(results_df)

# to print accuracy:
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
    
# To visualize model training and accuracy graph
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

#To visualize the model traing and validation loss graph
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

# Plot metrics comparison
# Color-blind friendly palette (Okabe-Ito)
cb_palette = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', 
              '#0072B2', '#D55E00', '#CC79A7', '#999999']

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
plt.figure(figsize=(15, 10))

for i, metric in enumerate(metrics):
    plt.subplot(2, 2, i+1)
    ax = sns.barplot(x=results_df.index, y=results_df[metric], palette=cb_palette)
    plt.title(f'{metric} Comparison')
    plt.xticks(rotation=45)
    
    # Add value labels on top of each bar
    for p in ax.patches:
        ax.annotate(
            f'{p.get_height():.2f}',  # Format to 2 decimal places
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='center', xytext=(0, 5), textcoords='offset points'
        )

plt.tight_layout()
plt.show()

# =============================================
# Confusion matrix
# =============================================

# Color-blind-friendly palette (replace 'Blues' with a diverging color map)
cb_cmap = sns.color_palette("rocket", as_cmap=True)  # Alternative: "rocket", "mako", or "crest"

y_true = np.argmax(y_test_onehot, axis=1)
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(12, 10))
    
    # Plot heatmap with smaller annotations
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap=cb_cmap,  
        annot_kws={"size": 8},  # Smaller annotation text
        xticklabels=range(num_classes), 
        yticklabels=range(num_classes),
        linewidths=0.5,  # Adds thin lines between cells
        linecolor='grey'  # Grid line color
    )
    
    plt.title(f'Confusion Matrix - {name}', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(fontsize=8)  # Smaller x-axis labels
    plt.yticks(fontsize=8)  # Smaller y-axis labels
    plt.show()
# =============================================
# Training Time Comparison (Balanced Style)
# =============================================
training_times = {
    "Basic CNN": 2510,  
    "Mini ResNet": 1824,
    "MinimobileNet": 1269,
    "LeNet":355
}

# Color-blind-friendly palette (Okabe-Ito)
cb_palette = ['#E69F00', '#56B4E9', '#009E73', '#0072B2', '#D55E00', '#CC79A7']

plt.figure(figsize=(8, 7))
ax = sns.barplot(
    x=list(training_times.keys()), 
    y=list(training_times.values()), 
    palette=cb_palette  # Apply color-blind-friendly colors
)

# Add values on top of bars
for p in ax.patches:
    ax.annotate(
        f'{int(p.get_height())}s',  # Format as "274s" (no decimals)
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center', va='bottom', xytext=(0, 5), textcoords='offset points',
        fontsize=10
    )

plt.title('Training Time Comparison', fontsize=14)
plt.ylabel('Time (seconds)', fontsize=12)
plt.xlabel('Model', fontsize=12)
plt.xticks(rotation=45, ha='right')  # Better angled labels
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add light gridlines for reference
plt.tight_layout()  # Prevent label cutoff
plt.show()

# ======================
# Model Size Comparison 
# ======================


# Calculate model sizes

model_sizes = {}
for name, filename in model_files.items():
    filepath = os.path.join(models_dir, filename)
    if os.path.exists(filepath):
        model_sizes[name] = os.path.getsize(filepath) / (1024 * 1024)  # Convert to MB
    else:
        print(f"Warning: File not found - {filepath}")

# Color-blind-friendly palette (Okabe-Ito)
cb_palette = ["#4477AA", "#66CCEE", "#228833", "#CCBB44", "#EE6677", "#AA3377"]

# Create the plot
plt.figure(figsize=(10, 7))
ax = sns.barplot(
    x=list(model_sizes.keys()), 
    y=list(model_sizes.values()),
    palette=cb_palette[:len(model_sizes)],  # Only use as many colors as needed
    edgecolor='black',
    linewidth=1,
    alpha=0.8  # Slightly transparent for better visibility
)

# Add MB values above each bar
for p in ax.patches:
    ax.annotate(
        f'{p.get_height():.1f} MB',  # Format to 1 decimal place
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center', va='bottom', 
        xytext=(0, 5), 
        textcoords='offset points',
        fontsize=10,
        weight='bold'  # Make text slightly bolder
    )

# Customize plot appearance
plt.title('Model Size Comparison (.keras files)', fontsize=14, pad=20)
plt.ylabel('Size (MB)', fontsize=12)
plt.xlabel('Model Architecture', fontsize=12)
plt.xticks(rotation=45, ha='right')  # Better angled labels

# Add grid and adjust layout
plt.grid(axis='y', linestyle='--', alpha=0.3)  # Light horizontal gridlines
ax.set_axisbelow(True)  # Grid behind bars
plt.tight_layout()

# Show the plot
plt.show()

# ======================
# To demonstrate an unseen image 
# ======================

# 1. Define paths and class names
models_dir = "C:/Users/LENOVO/Downloads/GRSRB/saved_models/"
test_image_path = "C:/Users/LENOVO/Downloads/traffic1.jpg" 

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

# 2. Load all saved models
model_files = {
    'CustomCNN': 'CustomCNN_traffic_sign_model.keras',
    'MiniResNet': 'MiniResNet_traffic_sign_model.keras',
    'MiniMobileNet': 'MiniMobileNet_traffic_sign_model.keras',
    'LeNet': 'LeNet_traffic_sign_model.keras'
}

models = {}
for name, file in model_files.items():
    try:
        models[name] = load_model(os.path.join(models_dir, file))
        print(f"✅ {name} loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load {name}: {str(e)}")


#3. Image preprocessing function
def preprocess_image(image_path, img_size=(28, 28)):
    img = Image.open(image_path).resize(img_size)
    img_array = np.array(img) / 255.0
    
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack((img_array,)*3, axis=-1)
    elif img_array.shape[2] == 4:  # RGBA
            img_array = img_array[:, :, :3]  # Remove alpha channel
               
    return np.expand_dims(img_array, axis=0)

# 4. Prediction and visualization
def predict_with_all_models(image_path):
    # Preprocess image
    try:
        img_array = preprocess_image(image_path)
        original_img = Image.open(image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Make predictions
    predictions = {}
    for name, model in models.items():
        pred = model.predict(img_array, verbose=0)
        pred_class = np.argmax(pred)
        confidence = np.max(pred)
        predictions[name] = {
            'class': pred_class,
            'name': class_names[pred_class],
            'confidence': confidence
        }
    
    # Visualization
    plt.figure(figsize=(15, 8))
    
    # Show original image
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title("Test Traffic Sign", fontsize=14)
    plt.axis('off')
    
    # Show predictions comparison
    plt.subplot(1, 2, 2)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(models)))
    
    for i, (name, pred) in enumerate(predictions.items()):
        plt.barh(name, pred['confidence']*100, color=colors[i], alpha=0.6)
        plt.text(pred['confidence']*100 + 1, i, 
                f"{pred['name']} ({pred['confidence']:.1%})",
                va='center', fontsize=10)
    
    plt.title('Model Predictions Comparison', fontsize=14)
    plt.xlabel('Confidence (%)', fontsize=12)
    plt.xlim(0, 100)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    return predictions

# 5. Run prediction on test image
if os.path.exists(test_image_path):
    predictions = predict_with_all_models(test_image_path)
    
    # Print detailed results
    print("\nDetailed Predictions:")
    for name, pred in predictions.items():
        print(f"{name}: {pred['name']} (Class {pred['class']}) - Confidence: {pred['confidence']:.2%}")
else:
    print(f"Test image not found at: {test_image_path}")

