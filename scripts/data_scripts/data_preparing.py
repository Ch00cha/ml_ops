import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

df = pd.read_csv('/content/smoking.csv', index_col = 'Unnamed: 0')
df = df.drop(['amt_weekends', 'amt_weekdays', 'type'], axis = 1)
labelencoder = LabelEncoder()
data_new = labelencoder.fit_transform(df.smoke)
df['smoke'] = data_new
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

smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(train_data, df['smoke'])