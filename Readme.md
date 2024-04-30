**Language Detection Project**
**Project Overview**

This project implements a language detection system that predicts the language of a given sentence or text input. It combines a machine learning model for prediction with a user-friendly graphical user interface (UI) built using Tkinter.

**Project Structure**

language.ipynb: This notebook contains the machine learning model training code. It's responsible for loading and preprocessing the training dataset, training various machine learning models (e.g., MLP, SVM, Naive Bayes), evaluating the models, and saving the trained model files.
main.py: This Python script implements the UI using Tkinter and interacts with the saved model files.

**Dependencies**

**For Training the Model:**
Python libraries: pandas, numpy, nltk, scikit-learn, tensorflow, keras
NLTK stopwords and punkt data

**For Running the UI:**
Python libraries: tkinter, pandas, numpy, re, tensorflow, keras, nltk
NLTK stopwords and punkt data (downloaded within the code)

**Running the Project**
**Training the Model:**
Open language.ipynb in a Jupyter Notebook environment or compatible IDE.
Run all code cells in the notebook. This will train the model, save the necessary files, and provide performance metrics.

**Running the UI:**
Ensure the trained model files (best_model_mlp.h5, label_encoder.pkl, tfidf_vectorizer.pkl) are in the same directory as main.py.

Run main.py using a Python interpreter ( python main.py).

**UI Functionality**

The UI provides a user-friendly way to interact with the trained language detection model. It includes the following elements:

Input Field: Users can enter a sentence for language prediction.
Prediction Button: Triggers the prediction process.
Clear Button: Clears the input field and prediction result.
Result Label: Displays the predicted language for the entered sentence.
Code Explanation

**The main.py script utilizes the Tkinter library to create the UI elements and handle user interactions. Key functions include:**

preprocess_text: Cleans and prepares the user input for model compatibility.
predict_language: Loads the saved model files, preprocesses the input, makes predictions, and displays the predicted language.
clear_input: Clears the user entry field and result label.

**Additional Notes**
The accuracy of language prediction depends on the quality of the training data and the chosen machine learning model.
This UI relies on a pre-trained model and supporting files generated by the separate training script.
Ensure all dependencies are installed before running the project.
Feel free to customize and enhance the project according to your needs!
