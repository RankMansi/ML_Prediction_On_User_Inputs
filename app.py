import os
from flask import Flask, render_template, request
from flask_cors import CORS
import numpy as np
from joblib import load
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import uuid

app = Flask(__name__)
CORS(app)

# Function to render the index.html template on GET request and process form data on POST request
@app.route('/', methods=['GET', 'POST'])
def hello_world():
    # Base path for the default image
    base_image_path = os.path.join("static", "base_pic.svg")
    
    if request.method == 'GET':
        # Render the index.html template with the base image path
        return render_template('index.html', href=base_image_path)
    elif request.method == 'POST':
        try:
            # Get text data from the form
            text = request.form['text']
            # Define the output file path
            output_file = os.path.join("static", f"prediction_{text}.svg")
            # Load the trained model
            model = load('model.joblib')
            # Convert input string to NumPy array of floats
            np_arr = floats_string_to_np_arr(text)
            # Generate the image based on input and save it
            make_picture('AgesAndHeights.pkl', model, np_arr, output_file)
            # Render the index.html template with the generated image
            return render_template('index.html', href=output_file)
        except Exception as e:
            # Render the index.html template with an error message if an exception occurs
            return render_template('index.html', href=base_image_path, error=str(e))

# Function to generate an interactive plot based on model predictions and input data
def make_picture(training_data_filename, model, new_inp_np_arr, output_file):
    try:
        if new_inp_np_arr is not None: 
            # Read the training data and filter out invalid ages
            data = pd.read_pickle(training_data_filename)
            ages = data['Age']
            data = data[ages > 0]
            ages = data['Age']
            heights = data['Height']
            x_new = np.array(list(range(19))).reshape(19, 1)
            preds = model.predict(x_new)

            # Create a plotly figure with scatter plot and model predictions
            fig = px.scatter(x=ages, y=heights, title="Height vs Age of People", labels={'x': 'Age (years)', 'y': 'Height (inches)'})
            fig.add_trace(go.Scatter(x=x_new.reshape(19), y=preds, mode='lines', name='Model'))

            # Predict new outputs and add them to the plot as markers
            new_inp_np_arr = new_inp_np_arr.reshape(len(new_inp_np_arr), 1)
            new_preds = model.predict(new_inp_np_arr)
            fig.add_trace(go.Scatter(x=new_inp_np_arr.reshape(len(new_inp_np_arr)), y=new_preds, name='New Outputs', mode='markers', marker=dict(color='purple', size=20, line=dict(color='purple', width=2))))

            # Save the figure as an image
            fig.write_image(output_file, width=800, engine='kaleido')
            return output_file
        else:
            return None
    except Exception as e:
        # Print an error message if an exception occurs during image generation
        print(f"An error occurred: {e}")
        return None

# Function to convert a string of floats to a NumPy array
def floats_string_to_np_arr(floats_str):
    # Check if a string can be converted to a float
    def is_float(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
    # Convert string of floats to a NumPy array
    floats = np.array([float(x) for x in floats_str.split(',') if is_float(x)])
    return floats.reshape(len(floats), 1)

# Start the Flask application
if __name__ == '__main__':
    app.run()
