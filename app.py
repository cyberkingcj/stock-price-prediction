import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import math
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', l=0)

@app.route('/visualize',methods=['POST'])
def visualize():
    return render_template('visualize.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = request.form['days']
    if(request.form['company'] == 'TCS'):
        model = pickle.load(open('model_TCS.pkl', 'rb'))
    elif(request.form['company'] == 'INFY'):
        model = pickle.load(open('model_INFY.pkl','rb'))
    int_features =int(int_features)
    y = np.exp(model.forecast(int_features)[0])
    y = np.round(y,2)
    if int_features == 1:
        output = y[0]
        return render_template('index.html', prediction_text='TCS stock price will be Rs. {}'.format(output), l=0)
    
    #prediction = np.exp(model.forecast(int_features)[0])
    prediction = y[int_features-1]
    #output = round(prediction, 2)
    nod = int_features
    dateparse = lambda dates: pd.datetime.strptime(dates, '%d-%m-%Y')
    dates = pd.read_csv('dates.csv', parse_dates=['Date'], date_parser=dateparse)
    x = [dates[i:i+nod] for i in range(0,dates.shape[0],nod)]
    x = pd.DataFrame(x)
    x = dates.loc[0:nod-1]
    x['Price'] = y
    fig = px.line(x, x='Date', y='Price')
    fig.write_html("templates/visualize.html")
    date = x.loc[nod-1]
    date = np.array(date)
    df = x
    s = pd.Series(i for i in range(1,nod+1))
    df.set_index(s)
    print(df)
    
    return render_template('index.html', prediction_text='TCS stock price will be Rs. {}'.format(prediction), tables=[df.to_html(classes='data')], titles=df.columns.values, vs_text = "Click here to visualize", l=1)

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    y = prediction[0]
    return jsonify(y)

if __name__ == "__main__":
    app.run(debug=True)