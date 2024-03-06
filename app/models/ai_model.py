from keras.models import load_model
import numpy as np
from PIL import Image
from keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
import pickle

class ImageCaptioningModel:
    def __init__(self, model_path, tokenizer_path):
        # Load the model
        self.model = load_model(model_path)
        
        # Load the tokenizer
        with open(tokenizer_path, 'rb') as tokenizer_file:
            self.tokenizer = pickle.load(tokenizer_file)
        
        # Xception model for feature extraction
        self.xception_model = Xception(include_top=False, pooling="avg")
    
    def extract_features(self, filename):
        try:
            image = Image.open(filename)
        except:
            print("ERROR: Couldn't open image! Make sure the image path and extension are correct")
        
        image = image.resize((299, 299))
        image = np.array(image)
        
        if image.shape[2] == 4:
            image = image[..., :3]
        
        image = np.expand_dims(image, axis=0)
        image = image / 127.5
        image = image - 1.0
        feature = self.xception_model.predict(image)
        return feature
    
    def word_for_id(self, integer):
        for word, index in self.tokenizer.word_index.items():
            if index == integer:
                return word
        return None
    
    def generate_desc(self, photo, max_length):
        in_text = 'start'
        for _ in range(max_length):
            sequence = self.tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=max_length)
            pred = self.model.predict([photo, sequence], verbose=0)
            pred = np.argmax(pred)
            word = self.word_for_id(pred)
            
            if word is None:
                break
            in_text += ' ' + word
            
            if word == 'end':
                break
        return in_text
    
    def predict_image_caption(self, image_path, max_length=51):
        photo = self.extract_features(image_path)
        img = Image.open(image_path)
        description = self.generate_desc(photo, max_length)
        return description

