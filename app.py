from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from config.db_connection import create_connection
import time
import numpy as np
import os
import mysql.connector
import json
import jwt
import datetime
from functools import wraps
from werkzeug.security import generate_password_hash
from werkzeug.security import check_password_hash, generate_password_hash

app = Flask(__name__)

# Secret key untuk menandatangani token JWT
app.config['SECRET_KEY'] = 'iki'

# Enable CORS for specified origins
CORS(app, origins=["http://localhost:3000", "http://your-frontend-url.com"])  # Replace with your actual frontend URL

# Inisialisasi koneksi database
db_connection = create_connection()

# Load the trained model
model = load_model('./hasil_latih/hasil_latihan_3/pest_classification_alexnet.h5')

# Function to preprocess the uploaded image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Resize to match model input
    img_array = img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values
    return img_array

# Function to insert prediction results into the database
def save_prediction_to_db(user_id, image_path, predicted_class, presentase_predicted_class, all_probabilities):
    try:
        cursor = db_connection.cursor()
        insert_query = """
            INSERT INTO Predictions (user_id, image_path, prediction_result, prediction_percentage, all_probabilities)
            VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (
            user_id, 
            image_path, 
            predicted_class, 
            presentase_predicted_class, 
            json.dumps(all_probabilities)  # Store probabilities as JSON
        ))
        db_connection.commit()
    except mysql.connector.Error as e:
        print(f"Error inserting data into database: {e}")
        db_connection.rollback()

# Middleware untuk memverifikasi token
def token_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = None

        # Memeriksa apakah token disertakan dalam header
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(" ")[1]  # Ambil token setelah "Bearer "

        if not token:
            return jsonify({'message': 'Token is missing!'}), 403

        try:
            # Verifikasi token
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = data['user_id']
        except:
            return jsonify({'message': 'Token is invalid!'}), 403

        return f(current_user, *args, **kwargs)
    return decorated_function

# Route untuk login dan menghasilkan token JWT
@app.route('/login', methods=['POST'])
def login():
    # Ambil data dari request body
    data = request.get_json()

    email = data.get('email')
    password = data.get('password')

    # Pastikan email dan password ada
    if not email or not password:
        return jsonify({'message': 'Email dan password diperlukan!'}), 400

    # Cari user berdasarkan email
    cursor = db_connection.cursor()
    cursor.execute("SELECT * FROM Users WHERE email = %s", (email,))
    user = cursor.fetchone()

    if user:
        # Cek apakah password sesuai
        if check_password_hash(user[3], password):  # user[3] adalah password yang di-hash
            # Jika user ditemukan dan password sesuai, buat JWT token
            token = jwt.encode({
                'user_id': user[0],  # user[0] adalah ID user
                'role': user[4],  # user[4] adalah role
                'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)
            }, app.config['SECRET_KEY'], algorithm='HS256')

            return jsonify({'token': token}), 200
        else:
            return jsonify({'message': 'Login failed! Incorrect email or password.'}), 401
    else:
        return jsonify({'message': 'User not found.'}), 404

# Route untuk upload gambar, dengan akses berbeda tergantung role

# In-memory storage for tracking guest requests (limited to 10 requests)
GUEST_REQUESTS = {}

# Max requests allowed for guests
MAX_GUEST_REQUESTS = 5
REQUEST_WINDOW = 3600  # 1 hour window (3600 seconds)

@app.route('/upload', methods=['POST'])
def upload_image():
    # Cek apakah user login atau tamu
    token = None
    if 'Authorization' in request.headers:
        token = request.headers['Authorization'].split(" ")[1]  # Ambil token setelah "Bearer "

    # Jika tidak ada token, anggap pengguna adalah tamu
    if not token:
        current_user = None
        role = "Tamu"
        # Get the user's IP address as the identifier for guest requests
        ip_address = request.remote_addr
        
        # Check if the IP has already made requests
        current_time = time.time()
        if ip_address in GUEST_REQUESTS:
            # Filter out requests that were made outside the request window (1 hour)
            GUEST_REQUESTS[ip_address] = [timestamp for timestamp in GUEST_REQUESTS[ip_address] if current_time - timestamp < REQUEST_WINDOW]
            
            # If the number of requests exceeds the limit, return an error
            if len(GUEST_REQUESTS[ip_address]) >= MAX_GUEST_REQUESTS:
                return jsonify({
                    'success': False, 
                    'message': 'Rate limit exceeded. Please try again later.',
                    'request_count': len(GUEST_REQUESTS[ip_address])  # Include request count in the response
                }), 429

        else:
            GUEST_REQUESTS[ip_address] = []

        # Record the current request time
        GUEST_REQUESTS[ip_address].append(current_time)

        # Include the current request count for the user
        request_count = len(GUEST_REQUESTS[ip_address])

    else:
        try:
            # Verifikasi token
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = data['user_id']
            role = data['role']
            request_count = 0  # No need to track requests for logged-in users
        except:
            return jsonify({'message': 'Token is invalid!'}), 403

    # Tamu tetap bisa mengakses prediksi, namun data tidak disimpan
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file uploaded.'}), 400
    
    file = request.files['file']
    
    if file and file.filename.endswith(('png', 'jpg', 'jpeg')):
        os.makedirs('uploads', exist_ok=True)
        
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        
        processed_image = preprocess_image(file_path)
        
        predictions = model.predict(processed_image)[0]
        
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
        
        predictions_dict = {class_name: f"{round(prob * 100, 2)}%" for class_name, prob in zip(class_names, predictions)}
        
        predicted_class_index = np.argmax(predictions)
        predicted_label = class_names[predicted_class_index]
        presentase_predicted_class = predictions_dict[predicted_label]
        
        # Jika role bukan 'Tamu', simpan prediksi ke database
        if current_user and role != "Tamu":
            save_prediction_to_db(current_user, file_path, predicted_label, presentase_predicted_class, predictions_dict)
        
        return jsonify({
            'success': True,
            'message': 'Image uploaded and predictions made successfully.',
            'request_count': request_count,  # Include request count in the response
            'data': { 
                'predicted_class': predicted_label,
                'presentase_predicted_class': presentase_predicted_class,
                'all_probabilities': predictions_dict
            }        
        }), 200
    else:
        return jsonify({'success': False, 'message': 'Invalid file type. Please upload a PNG, JPG, or JPEG image.'}), 400

    # Tamu tetap bisa mengakses prediksi, namun data tidak disimpan
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file uploaded.'}), 400
    
    file = request.files['file']
    
    if file and file.filename.endswith(('png', 'jpg', 'jpeg')):
        os.makedirs('uploads', exist_ok=True)
        
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        
        processed_image = preprocess_image(file_path)
        
        predictions = model.predict(processed_image)[0]
        
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
        
        predictions_dict = {class_name: f"{round(prob * 100, 2)}%" for class_name, prob in zip(class_names, predictions)}
        
        predicted_class_index = np.argmax(predictions)
        predicted_label = class_names[predicted_class_index]
        presentase_predicted_class = predictions_dict[predicted_label]
        
        # Jika role bukan 'Tamu', simpan prediksi ke database
        if current_user and role != "Tamu":
            save_prediction_to_db(current_user, file_path, predicted_label, presentase_predicted_class, predictions_dict)
        
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

@app.route('/register', methods=['POST'])
def register_user():
    # Ambil data dari request body
    data = request.get_json()

    # Pastikan data yang dibutuhkan ada
    if not data or not data.get('name') or not data.get('email') or not data.get('password'):
        return jsonify({'success': False, 'message': 'Data tidak lengkap!'}), 400

    name = data['name']
    email = data['email']
    password = data['password']
    role = data.get('role', 'User')  # Ambil role, default 'User' jika tidak ada

    # Cek apakah email sudah terdaftar
    cursor = db_connection.cursor()
    cursor.execute("SELECT * FROM Users WHERE email = %s", (email,))
    existing_user = cursor.fetchone()

    if existing_user:
        return jsonify({'success': False, 'message': 'Email sudah terdaftar!'}), 400

    # Enkripsi password
    hashed_password = generate_password_hash(password)

    # Simpan user baru ke dalam database
    try:
        cursor.execute(
            "INSERT INTO Users (name, email, password, role) VALUES (%s, %s, %s, %s)",
            (name, email, hashed_password, role)  # Role sekarang diambil dari input
        )
        db_connection.commit()
        return jsonify({'success': True, 'message': 'Registrasi berhasil!'}), 201
    except Exception as e:
        db_connection.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500

    
if __name__ == '__main__':
    app.run(debug=True, port=5001)


# ---------------------------------------------------------- #


# from flask import Flask, request, jsonify
# from flask_cors import CORS  # Import CORS
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from config.db_connection import create_connection
# import numpy as np
# import os

# app = Flask(__name__)

# # Enable CORS for specified origins
# CORS(app, origins=["http://localhost:3000", "http://your-frontend-url.com"])  # Replace with your actual frontend URL

# # Inisialisasi koneksi database
# db_connection = create_connection()

# # Load the trained model
# model = load_model('./hasil_latih/hasil_latihan_3/pest_classification_alexnet.h5')

# # Function to preprocess the uploaded image
# def preprocess_image(image_path):
#     # Load image
#     img = load_img(image_path, target_size=(224, 224))  # Resize to match model input
#     img_array = img_to_array(img)  # Convert image to array
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#     img_array /= 255.0  # Normalize pixel values
#     return img_array

# @app.route('/upload', methods=['POST'])
# def upload_image():
#     if 'file' not in request.files:
#         return jsonify({'success': False, 'message': 'No file uploaded.'}), 400
    
#     file = request.files['file']
    
#     # Check if the file is a valid image
#     if file and file.filename.endswith(('png', 'jpg', 'jpeg')):
#         # Create 'uploads' directory if it doesn't exist
#         os.makedirs('uploads', exist_ok=True)
        
#         file_path = os.path.join('uploads', file.filename)
#         file.save(file_path)  # Save the uploaded image
        
#         # Preprocess the image
#         processed_image = preprocess_image(file_path)
        
#         # Predict the probabilities for each class
#         predictions = model.predict(processed_image)[0]  # Get probabilities for each class
        
#         # Map predicted class index to class names
#         class_names = [
#             "asiatic_rice_borer",
#             "brown_plant_hopper",
#             "paddy_stem_maggot",
#             "rice_gall_midge",
#             "rice_leaf_caterpillar",
#             "rice_leaf_hopper",
#             "rice_leaf_roller",
#             "rice_shell_pest",
#             "rice_stem_fly",
#             "rice_water_weevil",
#             "thrips",
#             "yellow_rice_borer"
#         ]
        
#         # Convert probabilities to percentages and map them to class names
#         predictions_dict = {class_name: f"{round(prob * 100, 2)}%" for class_name, prob in zip(class_names, predictions)}
        
#         # Get the predicted class (with the highest probability)
#         predicted_class_index = np.argmax(predictions)
#         predicted_label = class_names[predicted_class_index]
#         presentase_predicted_class = predictions_dict[predicted_label]  # Percentage of the predicted class
        
#         # Return the formatted response
#         return jsonify({
#             'success': True,
#             'message': 'Image uploaded and predictions made successfully.',
#             'data': { 
#                 'predicted_class': predicted_label,
#                 'presentase_predicted_class': presentase_predicted_class,
#                 'all_probabilities': predictions_dict
#             }        
#         }), 200
#     else:
#         return jsonify({'success': False, 'message': 'Invalid file type. Please upload a PNG, JPG, or JPEG image.'}), 400

# @app.route('/test-db', methods=['GET'])
# def test_db():
#     if not db_connection:
#         return jsonify({'success': False, 'message': 'Tidak dapat terhubung ke database.'}), 500

#     try:
#         cursor = db_connection.cursor(dictionary=True)
#         cursor.execute("SELECT * FROM Users")  # Ganti dengan tabel Anda
#         users = cursor.fetchall()
#         return jsonify({'success': True, 'data': users}), 200
#     except Exception as e:
#         return jsonify({'success': False, 'message': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True, port=5001)  # Ganti 5001 dengan port lain yang belum digunakan
