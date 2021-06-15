from flask import Flask,render_template,request
import pandas as ps
import pickle
import os
import numpy as np
from sklearn.decomposition import PCA

app = Flask(__name__)
model=pickle.load(open('cc_fraud.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predicting():
    if request.method == 'POST':
        if request.files:
            dat=request.files['input']
            features=ps.read_csv(dat)
            pca = PCA(n_components=2)
            pca.fit(features)
            X = pca.transform(features)
            prediction=model.predict(X)

    return render_template('home.html', prediction_text=prediction)

if __name__ =='__main__':
    app.run(debug=True)
