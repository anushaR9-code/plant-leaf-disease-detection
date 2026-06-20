from flask import Flask, render_template, request, redirect, url_for, session, flash
import mysql.connector
import os
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import numpy as np
import cv2
import timm
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'plantleafconvnext_2025'
}

# Model configuration
MODEL_PATH = 'model/convnext_tiny_best.pth'
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Global variables for model
model = None
class_names = None
device = None

def get_db_connection():
    """Create and return a database connection"""
    return mysql.connector.connect(**DB_CONFIG)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load the ConvNeXt model"""
    global model, class_names, device
    
    if model is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        class_names = checkpoint['class_names']
        num_classes = len(class_names)
        
        # Create model
        model = timm.create_model('convnext_tiny', pretrained=False, num_classes=num_classes)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        
        print(f"Model loaded successfully on {device}")
        print(f"Number of classes: {num_classes}")

def preprocess_image_for_visualization(image):
    """Create grayscale, binary, and thresholded versions of the image"""
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Apply binary threshold
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Apply adaptive threshold
    thresholded = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
    
    return gray, binary, thresholded

def predict_image(image_path):
    """Predict disease from image"""
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Transform for model
    transform = T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)
        
    disease = class_names[pred_idx.item()]
    accuracy = confidence.item() * 100
    
    return disease, accuracy

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (name, email, password) VALUES (%s, %s, %s)",
                         (name, email, password))
            conn.commit()
            cursor.close()
            conn.close()
            
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except mysql.connector.IntegrityError:
            flash('Email already exists!', 'danger')
        except Exception as e:
            flash(f'Error: {str(e)}', 'danger')
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE email = %s AND password = %s",
                         (email, password))
            user = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if user:
                session['user_id'] = user['id']
                session['user_name'] = user['name']
                flash(f'Welcome back, {user["name"]}!', 'success')
                return redirect(url_for('predict'))
            else:
                flash('Invalid email or password!', 'danger')
        except Exception as e:
            flash(f'Error: {str(e)}', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """User logout"""
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page"""
    if 'user_id' not in session:
        flash('Please login first!', 'warning')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No image uploaded!', 'danger')
            return redirect(request.url)
        
        file = request.files['image']
        
        if file.filename == '':
            flash('No image selected!', 'danger')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Load model if not loaded
            load_model()
            
            # Predict
            disease, accuracy = predict_image(filepath)
            
            # Save to database
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO predictions (user_id, image_filename, disease_type, accuracy) VALUES (%s, %s, %s, %s)",
                    (session['user_id'], filename, disease, accuracy)
                )
                conn.commit()
                cursor.close()
                conn.close()
            except Exception as e:
                print(f"Database error: {e}")
            
            # Generate preprocessing images
            img = Image.open(filepath).convert('RGB')
            gray, binary, thresholded = preprocess_image_for_visualization(img)
            
            # Save preprocessing images
            gray_path = os.path.join(app.config['UPLOAD_FOLDER'], f"gray_{filename}")
            binary_path = os.path.join(app.config['UPLOAD_FOLDER'], f"binary_{filename}")
            thresh_path = os.path.join(app.config['UPLOAD_FOLDER'], f"thresh_{filename}")
            
            cv2.imwrite(gray_path, gray)
            cv2.imwrite(binary_path, binary)
            cv2.imwrite(thresh_path, thresholded)
            
            return render_template('result.html',
                                 original_image=filename,
                                 gray_image=f"gray_{filename}",
                                 binary_image=f"binary_{filename}",
                                 thresh_image=f"thresh_{filename}",
                                 disease=disease,
                                 accuracy=accuracy)
        else:
            flash('Invalid file type! Please upload PNG, JPG, or JPEG.', 'danger')
            return redirect(request.url)
    
    return render_template('predict.html')

@app.route('/history')
def history():
    """Prediction history"""
    if 'user_id' not in session:
        flash('Please login first!', 'warning')
        return redirect(url_for('login'))
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT * FROM predictions WHERE user_id = %s ORDER BY predicted_at DESC",
            (session['user_id'],)
        )
        predictions = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return render_template('history.html', predictions=predictions)
    except Exception as e:
        flash(f'Error: {str(e)}', 'danger')
        return redirect(url_for('predict'))

if __name__ == '__main__':
    app.run(debug=True)
