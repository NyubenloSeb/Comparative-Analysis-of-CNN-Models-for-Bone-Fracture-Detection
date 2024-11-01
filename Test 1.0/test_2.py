# Import necessary libraries
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50  # Use ResNet50 instead of VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report

# Define paths to dataset directories
train_dir = r'F:\Bone Frature Project\Test 1.0\dataset\train'
val_dir = r'F:\Bone Frature Project\Test 1.0\dataset\val'
test_dir = r'F:\Bone Frature Project\Test 1.0\dataset\testing'

# Step 1: Preprocess the Data using ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    rotation_range=20,  # Augmentation: random rotations
    width_shift_range=0.2,  # Random width shifts
    height_shift_range=0.2,  # Random height shifts
    zoom_range=0.2,  # Random zooms
    horizontal_flip=True,  # Random horizontal flips
)

# For validation and test sets, we only rescale images
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training, validation, and test data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Resize all images to 224x224
    batch_size=32,
    class_mode='binary',  # Binary classification: fracture vs. no fracture
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=8,
    class_mode='binary',
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=2,
    class_mode='binary',
    shuffle=False  # Do not shuffle test data
)

# Step 2: Create the Model using Transfer Learning
# Load the ResNet50 model with pretrained weights from ImageNet, without the top (classification) layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Use GlobalAveragePooling instead of Flatten
x = Dense(1024, activation='relu')(x)  # Fully connected layer
x = Dropout(0.5)(x)  # Dropout to prevent overfitting
predictions = Dense(1, activation='sigmoid')(x)  # Output layer for binary classification

# Combine the base model with the new custom layers
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the pre-trained base model (we will fine-tune later)
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Print a summary of the model architecture
model.summary()

# Step 3: Define Callbacks for Saving the Best Model and Early Stopping
callbacks = [
    ModelCheckpoint('resnet50_bone_fracture.keras', save_best_only=True, monitor='val_loss', mode='min'),
    EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)  # Stop if validation loss doesn't improve for 5 epochs
]

# Step 4: Train the Model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    callbacks=callbacks
)

# Step 5: Unfreeze some layers of the base model and fine-tune
for layer in base_model.layers[-10:]:  # Unfreeze the last 10 layers of ResNet50
    layer.trainable = True

# Recompile the model with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# Fine-tune the model
history_finetune = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=5,
    callbacks=callbacks
)
# Step 6: Plot Training and Validation Accuracy and Loss
def plot_training(history, history_finetune):
    acc = history.history['accuracy'] + history_finetune.history['accuracy']
    val_acc = history.history['val_accuracy'] + history_finetune.history['val_accuracy']
    loss = history.history['loss'] + history_finetune.history['loss']
    val_loss = history.history['val_loss'] + history_finetune.history['val_loss']

    epochs = range(1, len(acc) + 1)

    # Plot Accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()

plot_training(history, history_finetune)
# Step 6: Evaluate the Model on the Test Set
test_steps = (test_generator.samples + test_generator.batch_size - 1) // test_generator.batch_size
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_steps)
print(f'Test accuracy: {test_accuracy:.4f}')

# Step 7: Make Predictions on Test Data
y_pred = model.predict(test_generator, steps=test_steps)
y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions

# Generate confusion matrix
y_true = test_generator.classes
cm = confusion_matrix(y_true, y_pred)
print('Confusion Matrix:')
print(cm)

# Generate classification report
cr = classification_report(y_true, y_pred, target_names=['Not Fractured', 'Fractured'])
print('Classification Report:')
print(cr)
