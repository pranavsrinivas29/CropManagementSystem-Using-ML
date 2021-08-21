
from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
import fp


app = Flask(__name__)

sc=StandardScaler()
# Loading the saved decision tree model pickle
rf_pkl_filename = 'model1.pkl'
rf_pkl = open(rf_pkl_filename, 'rb')
rf_model = pickle.load(rf_pkl)
#print "Loaded Decision tree model :: ", decision_tree_model

#model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def hello_world():
    return render_template("home.html")


@app.route('/predict',methods=['POST'])
def predict():
    
    
    
    
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    #x = [[t,h,m,n,po,ph]]
    
   
    """prediction code"""
    fert=['Agroblen','DAP','GROMOR(28-28)','NPK(14-35-14)','NPK(20-20)']
    fr="Urea"
    
    
    #res=loaded_model.predict(x)
    #x = sc.transform(x)
    prediction=rf_model.predict(final)
    count=0
    c=""
    for i in range(0,5):
        if(prediction[0][i]==1):
            c=fert[i]
            count=count+1
            break
        i=i+1
            
    
    
    if count==1:
        
        crop=c
    else:
        crop=fr
        
    return render_template('home.html',pred=crop,inp=final,result=prediction)
    
   
    #return render_template('home.html',pred='Your Forest is in Danger.\nProbability of fire occuring is {}'.format(c))
    
@app.route('/cphome')
def hello():
    return render_template("cp.html")

if __name__ == '__main__':
    app.run(debug=True)