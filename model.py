# Importing the libraries
import pandas as pd
import pickle

dataset = pd.read_csv('housing_data.csv')

X = dataset.iloc[:,1:3] #area,bedrooms, bathrooms columns



y = dataset.iloc[:, 0] #price column

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 2200, 5]]))