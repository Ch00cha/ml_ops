import pandas as pd
import yaml
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json

test_data = pd.read_csv('./data/test.csv')
X_test = test_data.drop('smoke', axis = 1)
y_test = test_data['smoke']

model = pickle.load(open('models/model.pkl', 'rb'))
y_preds = model.predict(X_test)
score = accuracy_score(y_test, y_preds)
with open('evaluate/score.json', 'w') as f:
    json.dump({"score": score}, f)
