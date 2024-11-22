from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import pickle

#Loading Model
dtr_model=pickle.load(open('model.pkl','rb'))
preprocessor =pickle.load(open('preprocessor.pkl','rb'))

# create flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        Year = request.form['Year']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        pesticides_tonnes = request.form['pesticides_tonnes']
        avg_temp = request.form['avg_temp']
        Area = request.form['Area']
        Item = request.form['Item']

        if not all([Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]):
            error_message = "Please fill out all the fields before submitting."
            return render_template('index.html', prediction=None, error=error_message)

        input_data = (Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item)
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        transformed_feature = preprocessor.transform(input_data_reshaped)
        prediction = dtr_model.predict(transformed_feature)

        return render_template('index.html',prediction=int(prediction))

if __name__=='__main__':
    app.run(debug=True)