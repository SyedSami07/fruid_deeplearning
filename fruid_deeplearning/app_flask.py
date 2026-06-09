import os
import io
import traceback
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model

# Flask অ্যাপ এবং CORS ইনিশিয়ালাইজেশন
app = Flask(__name__)
CORS(app)

# কারেন্ট স্ক্রিপ্টের ডিরেক্টরি থেকে মডেলের পাথ সেট করা
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, "image_classifications.h5")

print(f"\n[DEBUG] Target Model Path: {MODEL_PATH}")
print(f"[DEBUG] Native exist check: {os.path.exists(MODEL_PATH)}\n")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Error: Model file not found at {MODEL_PATH}.")

# মডেল লোড করা
print("Loading Deep Learning Model Container...")
model = load_model(MODEL_PATH)
print("[SUCCESS] Model loaded perfectly!\n")

# ৩৬টি ফল ও সবজির ক্যাটাগরি লিস্ট
data_cat = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower',
    'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno',
    'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas',
    'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 
    'sweetpotato', 'tomato', 'turnip', 'watermelon'
]

img_height, img_width = 180, 180

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        print("[ERROR] No 'image' key found in request.files.")
        return jsonify({'error': 'No image file uploaded in form-data. Key must be named "image"'}), 400
        
    file = request.files['image']
    if file.filename == '':
        print("[ERROR] Filename is empty")
        return jsonify({'error': 'No selected file'}), 400

    try:
        print(f"\n[INFO] Received file: {file.filename}, Content Type: {file.content_type}")
        
        # 🟢 ফিক্স: Flask ফাইলকে বাইনারি স্ট্রিমে পড়া এবং io.BytesIO দিয়ে কেপাস করা
        image_bytes = file.read()
        image_stream = io.BytesIO(image_bytes)
        
        # ২. ইমেজ লোড এবং টার্গেট সাইজে রিসাইজ করা (180x180)
        img = tf.keras.utils.load_img(image_stream, target_size=(img_height, img_width))
        
        # ৩. ইমেজকে NumPy Array তে কনভার্ট করা
        img_array = tf.keras.utils.img_to_array(img)
        
        # ৪. ব্যাচ ডাইমেনশন যোগ করা (Shape becomes: [1, 180, 180, 3])
        img_batch = np.expand_dims(img_array, axis=0)

        print(f"[INFO] Processed image shape for model: {img_batch.shape}")

        # ৫. মডেল প্রেডিকশন
        prediction = model.predict(img_batch)
        
        # ৬. কনফিডেন্স স্কোর বের করা
        if hasattr(tf.nn, 'softmax'):
            confidence = tf.nn.softmax(prediction[0]).numpy()
        else:
            confidence = prediction[0]
        
        predicted_index = int(np.argmax(confidence))
        predicted_class = data_cat[predicted_index]
        confidence_score = float(confidence[predicted_index])

        print(f"[SUCCESS] Prediction: {predicted_class} ({confidence_score * 100:.2f}%)")

        return jsonify({
            'success': True,
            'label': predicted_class,
            'confidence': confidence_score
        })
        
    except Exception as e:
        print("\n" + "="*50)
        print("[CRITICAL ERROR INSIDE PREDICT ROUTE]")
        traceback.print_exc()
        print("="*50 + "\n")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)