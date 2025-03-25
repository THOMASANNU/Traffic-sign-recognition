# Traffic-sign-recognition

import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential, load_model
from collections import Counter
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from time import time
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import save_model

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
y_train_labels = np.argmax(y_train, axis=1)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)
class_weights = dict(enumerate(class_weights))

# Custom Model
custom_model = Sequential([
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

custom_model.compile(optimizer=RMSprop(learning_rate=0.0001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
# LeNet Model
lenet_model = Sequential([
    Conv2D(6, (5, 5), activation='relu', input_shape=(28, 28, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(16, (5, 5), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(120, activation='relu'),
    Dense(84, activation='relu'),
    Dense(num_classes, activation='softmax')
])

lenet_model.compile(optimizer=RMSprop(learning_rate=0.0001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

# Train and evaluate model
models = {
    "Custom Model": custom_model,
    "LeNet": lenet_model,
}

# Train and evaluate models
for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Training {name}...")
    print(f"{'='*50}")
    
    # Only apply class weights to custom model (or whichever models you want)
    if name == "Custom Model":
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            class_weight=class_weights,
            verbose=1
        )
    else:
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            verbose=1
        )
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{name} Test Accuracy: {test_accuracy * 100:.2f}%")
    
    # Save model
    save_path = f"C:/Users/LENOVO/Desktop/{name.replace(' ', '_')}_model.h5"
    model.save(save_path)
    print(f"Model saved to {save_path}")

print("\nTraining complete for all models!")

models = {
    "Custom Model": load_model("C:/Users/LENOVO/Desktop/Custom Model_model.h5"),
    "LeNet": load_model("C:/Users/LENOVO/Desktop/LeNet_model.h5"),
}

# 1. Load both saved models
lenet_model = load_model("C:/Users/LENOVO/Desktop/LeNet_model.h5")
custom_model = load_model("C:/Users/LENOVO/Desktop/Custom Model_model.h5")

# 2. Load and preprocess your test image (28x28 for both models)
image_path = "C:/Users/LENOVO/Downloads/traffic5.png"
img = Image.open(image_path).resize((28, 28))
img_array = np.array(img) / 255.0  # Normalize

# Handle grayscale and add batch dimension
if len(img_array.shape) == 2:
    img_array = np.stack((img_array,)*3, axis=-1)
img_array = np.expand_dims(img_array, axis=0)

# 3. Make predictions
lenet_pred = lenet_model.predict(img_array)
custom_pred = custom_model.predict(img_array)

# 4. Get results
def get_prediction_details(prediction, model_name):
    pred_class = np.argmax(prediction)
    confidence = np.max(prediction)
    return {
        'model': model_name,
        'class': pred_class,
        'confidence': f"{confidence:.2%}",
        'top3': np.argsort(prediction[0])[-3:][::-1]  # Get top 3 classes
    }

lenet_result = get_prediction_details(lenet_pred, "LeNet")
custom_result = get_prediction_details(custom_pred, "Custom CNN")

# 5. GTSRB Class Labels (first 10 shown as example - complete with all 43)
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

# 6. Visual comparison
plt.figure(figsize=(12, 5))

# Show image
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Test Image")
plt.axis('off')

# LeNet prediction
plt.subplot(1, 3, 2)
plt.barh(['LeNet'], [float(lenet_result['confidence'][:-1])], color='skyblue')
plt.xlim(0, 100)
plt.title(f"LeNet Prediction\n{class_names[lenet_result['class']]}\n{lenet_result['confidence']}")

# Custom CNN prediction
plt.subplot(1, 3, 3)
plt.barh(['Custom CNN'], [float(custom_result['confidence'][:-1])], color='lightgreen')
plt.xlim(0, 100)
plt.title(f"Custom CNN Prediction\n{class_names[custom_result['class']]}\n{custom_result['confidence']}")

plt.tight_layout()
plt.show()

# 7. Detailed comparison
print("\nDetailed Prediction Comparison:")
print(f"{'Model':<15} | {'Predicted Class':<20} | {'Confidence':<10} | {'Top 3 Classes'}")
print("-"*65)
for result in [lenet_result, custom_result]:
    top3_str = ", ".join([f"{c}:{class_names.get(c, '?')}" for c in result['top3']])
    print(f"{result['model']:<15} | {class_names.get(result['class'], 'Unknown'):<20} | {result['confidence']:<10} | {top3_str}")

# Make predictions
y_true = np.argmax(y_test, axis=1)
y_preds = {}
y_scores = {}

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_preds[name] = np.argmax(y_pred, axis=1)
    y_scores[name] = y_pred  # Store probabilities for ROC curve

# 1. Confusion Matrix Heatmaps
for name, y_pred in y_preds.items():
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(num_classes), 
                yticklabels=range(num_classes))
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# 2. Classification Report and Metrics Comparison
results = {}
for name, y_pred in y_preds.items():
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    results[name] = {
        "Precision": precision_score(y_true, y_pred, average='weighted'),
        "Recall": recall_score(y_true, y_pred, average='weighted'),
        "F1-Score": f1_score(y_true, y_pred, average='weighted'),
        "Accuracy": report['accuracy']
    }

results_df = pd.DataFrame(results).T
print("Performance Metrics:")
print(results_df)

# 3. ROC Curve Comparison
num_classes = y_test.shape[1]
y_test_bin = label_binarize(y_true, classes=np.arange(num_classes))
fpr, tpr, roc_auc = {}, {}, {}

for name in models.keys():
    fpr[name], tpr[name], _ = roc_curve(y_test_bin.ravel(), y_scores[name].ravel())
    roc_auc[name] = auc(fpr[name], tpr[name])

plt.figure(figsize=(10, 8))
for name in models.keys():
    plt.plot(fpr[name], tpr[name], label=f'{name} (AUC = {roc_auc[name]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.show()

metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
plt.figure(figsize=(12, 6))
for i, metric in enumerate(metrics):
    plt.subplot(2, 2, i+1)
    sns.barplot(x=results_df.index, y=results_df[metric])
    plt.title(f'{metric} Comparison')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

training_times = {
    "Custom Model": 110,  # Example (replace with actual training times)
    "LeNet": 46
}

plt.figure(figsize=(8, 5))
sns.barplot(x=list(training_times.keys()), y=list(training_times.values()))
plt.title('Training Time (Seconds)')
plt.ylabel('Time (s)')
plt.xticks(rotation=45)
plt.show()

model_sizes = {}
for name, model in models.items():
    model.save(f'temp_{name}.h5')
    model_sizes[name] = os.path.getsize(f'temp_{name}.h5') / (1024 * 1024)  # Convert to MB

plt.figure(figsize=(8, 5))
sns.barplot(x=list(model_sizes.keys()), y=list(model_sizes.values()))
plt.title('Model Size (MB)')
plt.ylabel('Size (MB)')
plt.xticks(rotation=45)
plt.show()

inference_times = {}
for name, model in models.items():
    start = time()
    model.predict(X_test[:1])  # Predict a single image
    inference_times[name] = (time() - start) * 1000  # Convert to milliseconds

plt.figure(figsize=(8, 5))
sns.barplot(x=list(inference_times.keys()), y=list(inference_times.values()))
plt.title('Inference Time (ms per Image)')
plt.ylabel('Time (ms)')
plt.xticks(rotation=45)
plt.show()
