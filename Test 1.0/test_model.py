# Import necessary libraries
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Step 1: Load the Trained Model
model = load_model('bone_fracture_detection.keras')

# Step 2: Prepare Input Data
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to match model input
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Scale pixel values to [0, 1]
    return img_array

# Step 3: Make Predictions on a Single Image
def predict_image(img_path):
    img_array = load_and_preprocess_image(img_path)
    prediction = model.predict(img_array)
    predicted_class = (prediction > 0.5).astype(int)
    
    return predicted_class[0][0]  # Return the predicted class

# Example Usage for a Single Image
# Replace 'path/to/your/image.jpg' with the actual image path
image_path = r'dataset/train/fractured/12-rotated2-rotated1-rotated3.jpg'
predicted_class = predict_image(image_path)

# Print the result
print(f"The model predicts: {'Fractured' if predicted_class == 1 else 'Not Fractured'}")
