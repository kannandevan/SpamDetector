from flask import Flask, request, render_template
import joblib

# Create the flask app
app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Define the homepage route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the message from the form
    message = request.form['message']
    
    # Convert the message into numbers using the saved vectorizer
    data = [message]
    vect = vectorizer.transform(data).toarray()
    
    # Make a prediction
    prediction = model.predict(vect)
    
    # Set the result text and style
    if prediction[0] == 1:
        result_text = "Result: This looks like SPAM!"
        result_class = "spam"
    else:
        result_text = "Result: This looks like a normal email (Ham)."
        result_class = "ham"

    # Render the page with the prediction result
    return render_template('index.html', prediction_text=result_text, prediction_class=result_class)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)