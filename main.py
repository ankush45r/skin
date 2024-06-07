from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
import numpy as np
import pickle
from keras.models import load_model

app = Flask(__name__)

# Load pre-trained model and scalers
model = load_model('./model.h5')

def preprocess_image(image):
    image = image.resize((28, 28))  # Resize to 28x28
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normalize pixel values
    return image_array

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('file.html', error='No file part')

        file = request.files['file']
        if file.filename == '':
            return render_template('file.html', error='No selected file')

        if file:
            try:
                image = Image.open(file.stream)
                processed_image = preprocess_image(image)
                processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension
                y_pred_probs = model.predict(processed_image)
                predicted_class = np.argmax(y_pred_probs)
                classes = {
                    0: 'Actinic Keratosis : A rough, scaly patch on the skin\ncaused by years of sun exposure. Actinic keratoses\nusually affect older adults. Reducing sun exposure can\nhelp reduce risk.',
                    1: 'Basal cell carcinoma : A type of skin cancer that beginsin\nthe basal cells. Basal cells produce new skin cells as old\nones die. Limiting sun exposure can help prevent these\ncells from becoming cancerous',
                    2: 'Benign Keratosis : A non-cancerous skin condition that\nappears as a waxy brown, black or tan growth. It usually\naffects older people. While it’s possible for one to appear\non its own, multiple growth are more common.',
                    3: 'Dermatofibroma : Dermatofibroma is a common\ncutaneous nodule of unknown etiology that occurs more\noften in women. Dermatofibroma frequently develops on\nthe extremities (mostly the lower legs) and is usually\nasymptomatic',
                    4: 'nv (Melanocytic Nevus : A usually non-cancerous disorderof pigment-producing skin cells commonly called birth\n marks or moles. This type of mole is often large and\ncaused by a disorder involving melanocytes, cells that\nproduce pigment.)',
                    5: 'Vascular Lesion : Vascular lesions are common\nabnormalities of the skin and underlying tissues, more\ncommonly known as birthmarks.',
                    6: 'Melanoma : Melanoma is the deadliest form of skin\ncancer. Melanoma occurs when the pigment-producing\ncells that give colour to the skin become cancerous.'
                }
                predicted_class_label = classes[predicted_class]
                return redirect(url_for('predict', predicted_class=predicted_class_label))
            except Exception as e:
                return render_template('file.html', error=str(e))

    return render_template('file.html')

@app.route("/predict", methods=['GET'])
def predict():
    predicted_class_label = request.args.get('predicted_class')
    return render_template('predict.html', predicted_class=predicted_class_label)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
