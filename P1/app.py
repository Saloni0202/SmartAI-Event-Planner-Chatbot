import os
import streamlit as st
import json
import random
import pickle
import string
import numpy as np
from datetime import datetime
import pandas as pd

# Load trained model, vectorizer & tag index
with open("chatbot_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("tag_to_index.pkl", "rb") as f:
    tag_to_index = pickle.load(f)

# Reverse tag_to_index mapping
index_to_tag = {i: tag for tag, i in tag_to_index.items()}

# Load intents.json
with open("intents.json", "r", encoding="utf-8") as file:
    intents = json.load(file)

# Preprocess text
def preprocess_text(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return text

# Get chatbot response
def get_response(user_input):
    user_input = preprocess_text(user_input)
    user_vector = vectorizer.transform([user_input])

    predicted_index = model.predict(user_vector)[0]
    predicted_tag = index_to_tag[predicted_index]

    print(f"Predicted tag: {predicted_tag}")  # Debugging output

    # Find responses from intents.json
    for intent in intents["intents"]:
        if intent["tag"] == predicted_tag:
            return random.choice(intent["responses"])

    return "I'm sorry, I didn't understand that. Can you rephrase?"

# Log chat history
def log_chat(user_input, bot_response):
    log_file = "chat_log.csv"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not os.path.isfile(log_file):
        df = pd.DataFrame(columns=["User Input", "Chatbot Response", "Timestamp"])
        df.to_csv(log_file, index=False)

    df = pd.DataFrame([[user_input, bot_response, timestamp]], columns=["User Input", "Chatbot Response", "Timestamp"])
    df.to_csv(log_file, mode='a', header=False, index=False)

# Streamlit UI
st.title("🎉 Event Planner Chatbot")
st.write("Ask me anything about planning an event! 🎈")

# User input
user_input = st.text_input("You:", "")

if user_input:
    response = get_response(user_input)
    st.write("🤖 Bot:", response)

    # Save chat history
    log_chat(user_input, response)