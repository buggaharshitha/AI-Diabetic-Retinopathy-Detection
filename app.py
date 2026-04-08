from flask import Flask, render_template, request, redirect, session
import numpy as np
from PIL import Image
import os
import tensorflow as tf

app = Flask(__name__)
app.secret_key = "secret123"   # required for login session

# LOGIN CREDENTIALS
USERNAME = "admin"
PASSWORD = "1234"

# LOAD MODEL (SavedModel)
MODEL_PATH = "final_model"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model folder not found at {MODEL_PATH}")

model = tf.saved_model.load(MODEL_PATH)
infer = model.signatures["serving_default"]

# CLASS LABELS
class_labels = [
    "Mild",
    "Moderate",
    "No DR",
    "Severe",
    "Proliferative DR"
]

# PREPROCESS
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)

    image = image / 127.5 - 1   # normalization

    image = np.expand_dims(image, axis=0)
    return image

# ROUTES
@app.route('/')
def index():
    if 'user' not in session:
        return redirect('/login')
    return render_template("index.html")


@app.route('/about')
def about():
    if 'user' not in session:
        return redirect('/login')
    return render_template("about.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user' not in session:
        return redirect('/login')

    if request.method == 'POST':

        file = request.files.get('file')

        if not file or file.filename == "":
            return render_template("prediction.html", error="Please select an image!")

        # Save file
        upload_dir = "static/uploads"
        os.makedirs(upload_dir, exist_ok=True)

        filepath = os.path.join(upload_dir, file.filename)
        file.save(filepath)

        # Open image
        image = Image.open(filepath).convert("RGB")
        processed_image = preprocess_image(image)

        # Prediction
        input_tensor = tf.convert_to_tensor(processed_image, dtype=tf.float32)

        pred_dict = infer(input_tensor)
        prediction_prob = list(pred_dict.values())[0].numpy()

        predicted_class = np.argmax(prediction_prob)
        confidence = np.max(prediction_prob) * 100

        result = class_labels[predicted_class]

        return render_template(
            "prediction.html",
            prediction=result,
            confidence=f"{confidence:.2f}%",
            filename=file.filename
        )

    return render_template("prediction.html")

# LOGIN
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = request.form['username']
        pwd = request.form['password']

        if user == USERNAME and pwd == PASSWORD:
            session['user'] = user
            return redirect('/')
        else:
            return render_template('login.html', error="Invalid credentials")

    return render_template('login.html')

# LOGOUT
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/login')

# RUN
if __name__ == "__main__":
    app.run(debug=True)
