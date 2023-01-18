import pickle
import json
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
#load the model
regmodel=pickle.load(open('regmodel.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data= scalar.transform(np.array(list(data.values())).reshape(1,-1))
    # new_data= scalar.transform([[-0.41709233,  0.29216419, -1.27338003, -0.28154625, -0.16513629,
    #      0.34715902, -0.13030059,  0.15267164, -0.97798895, -0.66777595,
    #     -1.32142483,  0.42854113, -1.04769976]])
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)
    print(output[0])
    return render_template("home.html",prediction_text="the House price pridiction is{}".format(output))



if __name__=="__main__":
    app.run(debug=True)
