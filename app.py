from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Enable CORS for specified origins
CORS(app, origins=["http://localhost:3000", "http://your-frontend-url.com"])  # Replace with your actual frontend URL

# Load the trained model
model = load_model('./hasil_latih/hasil_latihan_3/pest_classification_alexnet.h5')

# Function to preprocess the uploaded image
def preprocess_image(image_path):
    # Load image
    img = load_img(image_path, target_size=(224, 224))  # Resize to match model input
    img_array = img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values
    return img_array

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file uploaded.'}), 400
    
    file = request.files['file']
    
    # Check if the file is a valid image
    if file and file.filename.endswith(('png', 'jpg', 'jpeg')):
        # Create 'uploads' directory if it doesn't exist
        os.makedirs('uploads', exist_ok=True)
        
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)  # Save the uploaded image
        
        # Preprocess the image
        processed_image = preprocess_image(file_path)
        
        # Predict the probabilities for each class
        predictions = model.predict(processed_image)[0]  # Get probabilities for each class
        
        # Map predicted class index to class names
        class_names = [
            "asiatic_rice_borer",
            "brown_plant_hopper",
            "paddy_stem_maggot",
            "rice_gall_midge",
            "rice_leaf_caterpillar",
            "rice_leaf_hopper",
            "rice_leaf_roller",
            "rice_shell_pest",
            "rice_stem_fly",
            "rice_water_weevil",
            "thrips",
            "yellow_rice_borer"
        ]
        
        # Convert probabilities to percentages and map them to class names
        predictions_dict = {class_name: f"{round(prob * 100, 2)}%" for class_name, prob in zip(class_names, predictions)}
        
        # Get the predicted class (with the highest probability)
        predicted_class_index = np.argmax(predictions)
        predicted_label = class_names[predicted_class_index]
        presentase_predicted_class = predictions_dict[predicted_label]  # Percentage of the predicted class
        
        # Return the formatted response
        return jsonify({
            'success': True,
            'message': 'Image uploaded and predictions made successfully.',
            'data': { 
                'predicted_class': predicted_label,
                'presentase_predicted_class': presentase_predicted_class,
                'all_probabilities': predictions_dict
            }        
        }), 200
    else:
        return jsonify({'success': False, 'message': 'Invalid file type. Please upload a PNG, JPG, or JPEG image.'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Ganti 5001 dengan port lain yang belum digunakan
