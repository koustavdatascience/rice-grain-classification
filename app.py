from flask import Flask, request, render_template
from pathlib import Path
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Resolve paths from this file so local development and serverless execution use the same layout.
PROJECT_ROOT = Path(__file__).resolve().parent
UPLOAD_FOLDER = PROJECT_ROOT / 'static' / 'uploads'
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# TensorFlow and the model are optional at import time. This lets Vercel serve the project UI
# even though the full model is intentionally available only in the local development setup.
MODEL_PATH = PROJECT_ROOT / "rice_classifier_model.keras"
model = None
model_load_attempted = False
model_error = None

# Define class labels (ensure these match your training labels)
class_labels = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_model():
    """Load the local TensorFlow model only when an inference request is received."""
    global model, model_load_attempted, model_error
    if model_load_attempted:
        return model

    model_load_attempted = True
    if not MODEL_PATH.exists() or MODEL_PATH.read_text(errors="ignore").startswith("version https://git-lfs.github.com/spec"):
        model_error = "Live prediction is not enabled in this Vercel preview. Run the full project locally after downloading the trained model with Git LFS."
        return None

    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as exc:
        model_error = f"The local model could not be loaded: {exc}"
        return None


# Function to preprocess the input image
def preprocess_image(image_path):
    import numpy as np
    import tensorflow as tf
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    filename = None
    confidence = None
    error = None
    
    if request.method == "POST":
        # Check if the post request has the file part
        if 'image' not in request.files:
            error = "No file part"
        else:
            image_file = request.files["image"]
            # Check if the user selected a file
            if image_file.filename == '':
                error = "No selected file"
            elif image_file and allowed_file(image_file.filename):
                active_model = get_model()
                if active_model is None:
                    error = model_error
                    return render_template("index.html", prediction=None, filename=None, confidence=None, error=error)

                # Secure the filename to prevent security issues
                filename = secure_filename(image_file.filename)
                img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image_file.save(img_path)
                
                try:
                    # Preprocess the image and make a prediction
                    processed_image = preprocess_image(img_path)
                    import numpy as np
                    pred = active_model.predict(processed_image)
                    predicted_class = np.argmax(pred)
                    prediction = class_labels[predicted_class]
                    confidence = float(pred[0][predicted_class]) * 100
                except Exception as e:
                    error = f"Error processing image: {str(e)}"
                    filename = None
            else:
                error = "File type not allowed. Please upload a JPG, JPEG or PNG file."
    
    # Render the home page with the form and, if available, the prediction
    return render_template(
        "index.html", 
        prediction=prediction, 
        filename=filename,
        confidence=confidence,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)
