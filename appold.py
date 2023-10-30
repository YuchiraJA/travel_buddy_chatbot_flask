import json
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

# Load the trained model
with open('chatbot.dat', 'rb') as f:
    model = pickle.load(f)

# API endpoint to handle chatbot requests
@app.route('/predict/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()  # Get the input data from the POST request
    text_data = data['question']  # Extract the 'question' from the input data

    # Make prediction using the pre-trained model
    predicted_answer = model.predict([text_data])[0]

    # Return the predicted answer as a JSON response
    response = {'answer': predicted_answer}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5011)
