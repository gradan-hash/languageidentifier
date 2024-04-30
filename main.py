import pickle
import re
import tkinter as tk
from tkinter import ttk

import nltk
import numpy as np
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model


#to install dependencies use pip:

#example pip install tensorflow
# Function for text preprocessing
def preprocess_text(text):
    nltk.download('stopwords')
    nltk.download('punkt')

    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text)
    filtered_text = [word for word in tokens if word not in stop_words]
    processed_text = " ".join(filtered_text)
    return processed_text

# Function to predict language
def predict_language():
    try:
        # Load the saved TF-IDF vectorizer
        with open('./tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)

        # Load the saved MLP model
        model = load_model('./best_model_mlp.h5')

        # Load the label encoder used during training
        with open('./label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)

        # Get user input
        user_input = user_entry.get()

        if user_input:
            # Preprocess the input
            processed_input = preprocess_text(user_input)

            # TF-IDF Vectorization
            X_input = tfidf_vectorizer.transform([processed_input])

            # Convert the sparse matrix to a TensorFlow SparseTensor
            X_input_coo = X_input.tocoo()
            X_input_sparse_tensor = tf.sparse.SparseTensor(
                indices=np.vstack((X_input_coo.row, X_input_coo.col)).T,
                values=X_input_coo.data,
                dense_shape=X_input_coo.shape
            )

            # Reorder the sparse tensor indices if necessary
            X_input_reordered = tf.sparse.reorder(X_input_sparse_tensor)

            # Predict the language index
            prediction_scores = model.predict(X_input_reordered)
            prediction = np.argmax(prediction_scores, axis=-1)

            # Check if the prediction is within the range of known classes
            if prediction[0] < len(label_encoder.classes_):
                predicted_language = label_encoder.inverse_transform([prediction[0]])
                result_label.config(text=f'Predicted Language: {predicted_language[0]}')
            else:
                result_label.config(text='Error: Language does not exist in the dataset.')
        else:
            result_label.config(text='Please enter a sentence.')
    except Exception as e:
        result_label.config(text='Error: Language not recognized or is not in the dataset.')
        print(f'Error: {e}')






# Function to clear the input field and result
def clear_input():
    user_entry.delete(0, tk.END)
    result_label.config(text="")

# Create main window
root = tk.Tk()
root.title("Language Detection")

# Create input label and entry
input_label = ttk.Label(root, text="Enter a sentence to predict its language:")
input_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
user_entry = ttk.Entry(root, width=70)
user_entry.grid(row=0, column=1, padx=10, pady=10)

# Create predict button
predict_button = ttk.Button(
    root, text="Predict Language", command=predict_language)
predict_button.grid(row=0, column=2, padx=10, pady=10)

# Create clear button
clear_button = ttk.Button(
    root, text="Clear", command=clear_input)
clear_button.grid(row=0, column=3, padx=10, pady=10)

# Create result label
result_label = ttk.Label(root, text="")
result_label.grid(row=1, column=0, columnspan=4, padx=10, pady=10)

# Make the window resizable
root.resizable(True, True)

# Run the application
root.mainloop()
