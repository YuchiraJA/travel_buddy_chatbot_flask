import json
import pickle
import openai  # Import the OpenAI library
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

# Load the trained model
with open('chatbot.dat', 'rb') as f:
    model = pickle.load(f)

# Set up your OpenAI API key
openai.api_key = "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"  # Replace with your actual OpenAI API key
#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
# Function to use the pre-trained model
def use_trained_model(question):
    predicted_answer = model.predict([question])[0]
    return predicted_answer

# Function to use the OpenAI API
def use_openai_api(question):
    response = openai.Completion.create(
        engine="text-davinci-002",  # Choose an engine based on your requirements
        prompt=question,
        max_tokens=50  # Adjust the response length as needed
    )
    return response.choices[0].text

# API endpoint to handle chatbot requests
@app.route('/predict/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()  # Get the input data from the POST request
    text_data = data['question']  # Extract the 'question' from the input data

    # Check if the question is in the trained model
    if text_data in model.classes_:
        predicted_answer = use_trained_model(text_data)
    else:
        predicted_answer = use_openai_api(text_data)

    # Return the predicted answer as a JSON response
    response = {'answer': predicted_answer}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5011)
