from flask import Flask, jsonify, request
from flask_cors import CORS
from app.models.ai_model import ImageCaptioningModel

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

image_caption_model = None  # Initialize the model globally

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Perform AI model prediction
    predictions = image_caption_model.predict_image_caption(file)

    # Process predictions and return the results
    # (Adjust this based on your model and application requirements)

    return jsonify({'predictions': predictions})

def create_image_captioning_model():
    global image_caption_model
    model_path = 'app/models/model_0.h5'
    tokenizer_path = 'app/models/tokenizer.pkl'
    image_caption_model = ImageCaptioningModel(model_path, tokenizer_path)

if __name__ == '__main__':
    create_image_captioning_model()
    app.run(debug=True)
