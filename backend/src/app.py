from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model and tokenizer
model = tf.keras.models.load_model("/app/model/fake_news_detection_model_BILSTM.h5")
with open("/app/tokenizer.pickle", 'rb') as handle:
    tokenizer = pickle.load(handle)

# Predict function
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    # Preprocess text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=200)
    # Make prediction
    prediction = model.predict(padded_sequence)
    label = 1 if prediction[0][0] > 0.5 else 0  # 1 for true, 0 for fake
    return jsonify({'prediction': label, 'probability': float(prediction[0][0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # Make sure Flask listens on all interfaces
