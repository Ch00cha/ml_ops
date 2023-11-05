import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import yaml

df = pd.read_csv('./data/raw/smoking.csv', index_col = 'Unnamed: 0')
df = df.drop(['amt_weekends', 'amt_weekdays', 'type'], axis = 1)
labelencoder = LabelEncoder()
data_new = labelencoder.fit_transform(df.smoke)
df['smoke'] = data_new
cat_columns = []
num_columns = []

for column_name in df.columns:
    if (df[column_name].dtypes == object):
        cat_columns +=[column_name]
    else:
        num_columns +=[column_name]
# колонка национальностей
del_nationalities = ['Welsh', 'Irish', 'Refused', 'Unknown']
for nat in del_nationalities:
    df['nationality'] = np.where( (df['nationality'] == nat),
                                           'Other',
                                           df['nationality']
                                    )
#Колонка этнической принадлежности
ethnicities = ['Unknown', 'Refused', 'Mixed', 'Chinese', 'Black', 'Asian']
for ethnicity in ethnicities:
    df['ethnicity'] = np.where( (df['ethnicity'] == ethnicity),
                                           'Other',
                                           df['ethnicity']
                                    )
# Колонка дохода
for ethnicity in ethnicities:
    df['gross_income'] = np.where( (df['gross_income'] == 'Unknown'),
                                           'Refused',
                                           df['gross_income']
                                    )
encoded_data = pd.get_dummies(df[cat_columns])
data = pd.concat([encoded_data, df[num_columns[0]]], axis=1)
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(data, df['smoke'])

params_split = yaml.safe_load(open("params.yaml"))["split"]
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size = params_split["split_ratio"], random_state = params_split["seed"])
train_data = pd.concat([X_train, y_train], axis = 1)
test_data = pd.concat([X_test, y_test], axis = 1)
train_data.to_csv('data/train.csv', index=False)
test_data.to_csv('data/test.csv', index=False)


