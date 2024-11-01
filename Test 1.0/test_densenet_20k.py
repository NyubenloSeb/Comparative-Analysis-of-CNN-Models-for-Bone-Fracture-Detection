import os
import pandas as pd
import matplotlib.pyplot as plt  # Import matplotlib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report

# Define paths to dataset directories
train_image_dir = r'F:\Bone Frature Project\Test 1.0\YOLODataSet\images\train'
val_image_dir = r'F:\Bone Frature Project\Test 1.0\YOLODataSet\images\val'
train_label_dir = r'F:\Bone Frature Project\Test 1.0\YOLODataSet\labels\train'
val_label_dir = r'F:\Bone Frature Project\Test 1.0\YOLODataSet\labels\val'

# Define a mapping from class name to integer label (update if necessary)
class_mapping = {
    'XR_ELBOW': 0,
    'XR_FINGER': 1,
    'XR_FOREARM': 2,
    'XR_HAND': 3,
    'XR_SHOULDER': 4,
}

# Function to load labels from .txt files
def load_labels_from_txt(label_dir, image_dir):
    labels = []
    image_paths = []

    # Iterate over each label file in the directory
    for filename in os.listdir(label_dir):
        if filename.endswith('.txt'):
            # Extract class label from filename
            class_key = filename.split('_')[0] + '_' + filename.split('_')[1]  # Get the class name
            class_label = class_mapping.get(class_key)  # Map to the integer label

            if class_label is not None:
                # Get corresponding image path
                base_name = os.path.splitext(filename)[0]  # Remove .txt extension
                image_path = os.path.join(image_dir, base_name + '.png')

                if os.path.isfile(image_path):  # Check if the image file exists
                    labels.append(class_label)
                    image_paths.append(image_path)

    return pd.DataFrame({'image_path': image_paths, 'class_label': labels})

# Load training and validation labels
train_df = load_labels_from_txt(train_label_dir, train_image_dir)
val_df = load_labels_from_txt(val_label_dir, val_image_dir)

# Print DataFrame shapes and a few entries for validation
print(f'Loaded Training DataFrame:\n{train_df.head()}')
print(f'Training DataFrame shape: {train_df.shape}')
print(f'Loaded Validation DataFrame:\n{val_df.head()}')
print(f'Validation DataFrame shape: {val_df.shape}')

# Check for valid image paths
print("Valid training image paths:")
print(train_df['image_path'][train_df['image_path'].apply(os.path.isfile)])

print("Valid validation image paths:")
print(val_df['image_path'][val_df['image_path'].apply(os.path.isfile)])

# Convert class labels to string for multi-class classification
train_df['class_label'] = train_df['class_label'].astype(str)
val_df['class_label'] = val_df['class_label'].astype(str)

# Define ImageDataGenerator with data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

# For validation set, only rescale images
val_datagen = ImageDataGenerator(rescale=1. / 255)

# Create data generators
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='image_path',
    y_col='class_label',
    target_size=(224, 224),
    class_mode='categorical',  # Use 'categorical' for multi-class
    batch_size=32,
    shuffle=True
)

validation_generator = val_datagen.flow_from_dataframe(
    val_df,
    x_col='image_path',
    y_col='class_label',
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=8,
)

# Step 2: Create the Model using Transfer Learning
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers on top of the base model
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(len(class_mapping), activation='softmax')(x)  # Set output layer to the number of classes based on the mapping

# Combine the base model with the new custom layers
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the pre-trained base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Print a summary of the model architecture
model.summary()

# Step 3: Define Callbacks for Saving the Best Model and Early Stopping
callbacks = [
    ModelCheckpoint('densenet_bone_fracture.keras', save_best_only=True, monitor='val_loss', mode='min'),
    EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)
]

# Step 4: Train the Model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    callbacks=callbacks
)

# Step 6: Evaluate the Model on the Validation Set
val_steps = (validation_generator.samples + validation_generator.batch_size - 1) // validation_generator.batch_size
val_loss, val_accuracy = model.evaluate(validation_generator, steps=val_steps)
print(f'Validation accuracy: {val_accuracy:.4f}')

# Step 7: Make Predictions on Validation Data
y_pred = model.predict(validation_generator, steps=val_steps)
y_pred_classes = y_pred.argmax(axis=1)  # Get class with highest probability

# Generate confusion matrix and classification report
y_true = validation_generator.classes
cm = confusion_matrix(y_true, y_pred_classes)
print('Confusion Matrix:')
print(cm)

cr = classification_report(y_true, y_pred_classes, target_names=list(class_mapping.keys()))
print('Classification Report:')
print(cr)

# Step 8: Plot Training and Validation Accuracy and Loss
# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy: DenseNet121')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss: DenseNet121')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()
