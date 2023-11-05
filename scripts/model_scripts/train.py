from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import yaml
import pickle

params_train = yaml.safe_load(open("params.yaml"))["train"]
train_data = pd.read_csv('./data/train.csv')
y_train = train_data['smoke']
X_train = train_data.drop('smoke', axis = 1)
clf = RandomForestClassifier(max_depth = params_train["max_depth"],
                             criterion = params_train["criterion"],
                             min_samples_split = params_train["min_samples_split"],
                             min_samples_leaf = params_train["min_samples_leaf"])
clf.fit(X_train, y_train)

pickle.dump(clf, open('models/model.pkl', 'wb'))