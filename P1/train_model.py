import json
import string
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Load intents.json
with open("intents.json", "r", encoding="utf-8") as file:
    intents = json.load(file)

# Preprocess text: Lowercasing & Removing Punctuation
def preprocess_text(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return text

# Extract patterns and tags
patterns = []
tags = []
tag_to_index = {}

for intent in intents["intents"]:
    tag = intent["tag"]
    if tag not in tag_to_index:
        tag_to_index[tag] = len(tag_to_index)
    for pattern in intent["patterns"]:
        patterns.append(preprocess_text(pattern))
        tags.append(tag_to_index[tag])

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)
y = np.array(tags)

# Split data into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an SVM model
model = SVC(kernel="linear", probability=True)
model.fit(X_train, y_train)

# Save the model & vectorizer
with open("chatbot_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("tag_to_index.pkl", "wb") as f:
    pickle.dump(tag_to_index, f)

print("✅ Model training complete! Saved as chatbot_model.pkl")