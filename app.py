import joblib
import math
import pandas as pd
from flask import Flask, request, jsonify

# Initialize Flask app with a name
app = Flask("SuperKart Sales Predictor")

# Load the trained prediction model
model = joblib.load("superkart_prediction_model_v1_0.joblib")

# Define a route for the home page
@app.get('/')
def home():
    return "Welcome to the SuperKart Sales Prediction API!"

# Define an endpoint to predict churn for a single product
@app.post('/v1/product')
def predict_sales():
    # Get JSON data from the request
    product_data = request.get_json()

    # Extract relevant product features from the input data
    sample = {
        'Product_Weight': product_data['Product_Weight'],
        'Product_Sugar_Content': product_data['Product_Sugar_Content'],
        'Product_Allocated_Area': product_data['Product_Allocated_Area'],
        'Product_Type': product_data['Product_Type'],
        'Product_MRP': product_data['Product_MRP'],
        'Store_Id': product_data['Store_Id'],
        'Store_Establishment_Year': product_data['Store_Establishment_Year'],
        'Store_Size': product_data['Store_Size'],
        'Store_Location_City_Type': product_data['Store_Location_City_Type'],
        'Store_Type': product_data['Store_Type'],
    }

    # Convert the extracted data into a DataFrame
    input_data = pd.DataFrame([sample])

    # Make a sales prediction using the trained model
    result = model.predict(input_data).tolist()[0]
    prediction = math.ceil(result * 100) / 100

    # Return the prediction as a JSON response
    return jsonify({'Prediction': prediction})

# Define an endpoint to predict sales for a batch of products
@app.post('/v1/productbatch')
def predict_sales_batch():
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the file into a DataFrame
    input_data = pd.read_csv(file)
    
    # Preprocess any data for the model
    input_data['Store_Establishment_Year'] = input_data['Store_Establishment_Year'].astype(object)

    # Make predictions for the batch data and convert raw predictions into a readable format
    predictions = [
        math.ceil(x * 100) / 100
        for x in model.predict(input_data.drop("Product_Id",axis=1)).tolist()
    ]

    prod_id_list = input_data.Product_Id.values.tolist()
    output_dict = dict(zip(prod_id_list, predictions))

    return output_dict

# Run the Flask app in debug mode
if __name__ == '__main__':
    app.run(debug=True)
