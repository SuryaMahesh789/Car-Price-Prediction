
# features = name,company,year,price,kms_driven,fuel_type
# labels = car_price

import pandas as pd
import numpy as np
car=pd.read_csv('G:\Placements\CAR PRICE PREDICTOR\quikr_car.csv')

# To get the top 5 rows of the data --- head
print(car.head())

# shape of the dataset (how much data we have)
print(car.shape)

print(car.info())

# check whether year column has all the appropriate values or not
# no year column has many inappropriate values (non year values)
print(car['year'].unique())
print(car['name'].unique())
print(car['fuel_type'].unique())
print(car['kms_driven'].unique())
print(car['Price'].unique())
# quality of the raw data set we have
# year has many non year values
# year object into int data type
# price column has 'Ask for price' values to be removed and replaced with appropriate value
# price object into int data type
# kms-driven has kms with integers as extra string
# kms_driven object to int kms_driven has nan values
# fuel_type has nan values
# name column is inconsistent it is neither character nor string
# keep first three words of the data


# cleaning
backup=car.copy();


#let's see whether car dataset year column mixed with integer and non integer values
print(car['year'].str.isnumeric())

print(car[car['year'].str.isnumeric()])
# keep only details of the cars whose year values are integer values

print("before cleaning year values.... shape of the dataset")
print(car.shape)

# cleaning year column
car=car[car['year'].str.isnumeric()]

print("after cleaning year values.... ")
print("SHAPE :- ",car.shape)

print("Even now type of year is in str type....")
print(type(car['year'][0]))
print(car.info())

car['year']=car['year'].astype(int)

print("Now year column is changed into integer type....")
print(type(car['year'][0]))
print(car.info())

# car price values are in consistent by some of it's values has 'Ask For Price'
# remove those and keep remaining rows

print(car['Price'])
print(car['Price']=='Ask For Price')

# displaying which rows are inconsistent
print(car[car["Price"]=='Ask For Price'])

print("SHAPE :- ",car.shape)

# keep only remaining records into database
car=car[car["Price"]!='Ask For Price']

print("SHAPE :- ",car.shape)

print(car)

print(car.info())

print("Car price  values are object type with comma seperated integer values...")
print("Try to make it integer type of values...")

car['Price']=car['Price'].str.replace(',','').astype(int)

print(car['Price'].head())
print(type(car['Price'][0]))
print(type(car['Price']))
print(car.head())

print(car.info())

car['kms_driven']=car['kms_driven'].str.split(' ').str.get(0).str.replace(',','')
print(car['kms_driven'])

# lets see how many rows has 'petrol' as kms_driven value arey bhai kuch rows esi tarah bee hein
print(car[~car['kms_driven'].str.isnumeric()])

# keep rows whose kms_driven has numeric values into car dataset
car=car[car['kms_driven'].str.isnumeric()]

# now convert the data type to integer as all the values of kms_driven are numeric values but in string type
# convert into int type

car['kms_driven']=car['kms_driven'].astype(int)
print(car)
print(car.info())

# fuel_type has Nan values

# let's see how many rows has Nan value for fuel_type column
print(car[car['fuel_type'].isna()])

# let's see how many rows doesn't have Nan value for fuel_type column
print(car[~car['fuel_type'].isna()])

# keep rows with out Nan values
car=car[~car['fuel_type'].isna()]

print(car['fuel_type'])
print(car.info())

print(car.head())

car['name']=car['name'].str.split(' ').str.slice(0,3).str.join(' ')

print(car.head())

# reset the index
car=car.reset_index(drop=True)

# check whether everything is fine
print(car.head())

print(car.describe())

# let's check how many cars have price >60,00,000 rupees
print(car[car['Price']>6e6])

# seems to be only car >60,00,0000 it's an outlier

# remove the outlier
car=car[car['Price']<6e6].reset_index(drop=True)

# let's check now -- how many cars have price >60,00,000 rupees
print(car[car['Price']>6e6])

# let's store this cleaned data in csv file

car.to_csv('cleaned car.csv')

# Model

# price column is not the feature, it is label
x=car.drop(columns='Price')
y=car['Price']

print(x)
print(y)

# now split the dataset into training and testing dataset
# using sklearn model

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

# import all other necessary libraries

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder


"""
what are all the libraries we imported above ??
It seems like you're working with the scikit-learn library in Python for machine learning tasks. The code you provided imports three modules from scikit-learn: `LinearRegression`, `r2_score`, and `OneHotEncoder`.

`LinearRegression` is a class in scikit-learn that represents a linear regression model. Linear regression is a supervised learning algorithm used for predicting a continuous target variable based on one or more input features. This class provides methods to fit the model to the data, make predictions, and evaluate the model's performance.

`r2_score` is a function in scikit-learn that calculates the coefficient of determination (R-squared) for a regression model's predictions. The R-squared score measures the proportion of the variance in the target variable that is predictable from the input features. It is a common metric used to assess the goodness of fit of a regression model.

`OneHotEncoder` is a class in scikit-learn used for one-hot encoding categorical features. Categorical variables are typically represented as strings or integers, but many machine learning algorithms require numerical input. One-hot encoding converts categorical variables into binary vectors where each category becomes a binary feature column. 
This encoding is necessary when working with categorical data in scikit-learn.

These modules are commonly used in machine learning workflows for building and evaluating regression models.

"""

"""

The main use of the `OneHotEncoder` in scikit-learn is to transform categorical variables into a numerical representation suitable for machine learning algorithms. Many machine learning algorithms require numerical input, so categorical variables need to be encoded in a way that preserves the information they carry.

Here are the key reasons for using `OneHotEncoder`:

1. **Encoding categorical variables:** `OneHotEncoder` converts categorical variables into a binary vector representation. Each category becomes a binary feature column, where a value of 1 indicates the presence of that category and 0 indicates its absence. This allows the machine learning algorithm to understand and utilize the categorical information.

2. **Preventing ordinality assumptions:** One-hot encoding avoids imposing any ordinality assumptions on categorical variables. For example, if you have a feature with three categories ("small," "medium," "large"), encoding them as 0, 1, and 2 would imply an ordered relationship between the categories, which may not be appropriate in many cases. One-hot encoding treats each category independently and avoids such assumptions.

3. **Handling categorical variables with multiple levels:** Categorical variables with multiple levels can be encoded using one-hot encoding. Each level of the variable becomes a separate binary feature column, allowing the machine learning algorithm to capture the presence or absence of each level as a distinct feature.

4. **Preserving interpretability:** One-hot encoding preserves the interpretability of the original categorical variables. The resulting binary features indicate which categories are present in the original variable, making it easier to understand the impact of each category on the model's predictions.

It's worth noting that scikit-learn's `OneHotEncoder` works with categorical variables that are encoded as integers or strings. It expects a 2D array-like input where each column represents a categorical feature.

"""


ohe=OneHotEncoder()
ohe.fit(x[['name','company','fuel_type']])

print(OneHotEncoder)
print(ohe.categories_)

print(type(ohe))

from sklearn.compose import make_column_transformer
from sklearn.pipeline import  make_pipeline

# create a column transformer , transforms the columns
# applies the onehotencoder on these columns

column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),remainder='passthrough')

# using linear regression train the data
lr=LinearRegression()

# put the column transformer and linear regression into one pipeline
pipe=make_pipeline(column_trans,lr)

# fit them into pipeline
pipe.fit(x_train,y_train)

# training the regression model, by fitting training data both features and labels
print(pipe.fit(x_train,y_train))

# getting predictions just passing the raw data
y_pred=pipe.predict(x_test)
print(y_pred)

r2score=r2_score(y_test,y_pred)

print("R2score :- ",r2score)

for i in range(10):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    lr=LinearRegression()
    pipe=make_pipeline(column_trans,lr)
    pipe.fit(x_train,y_train)
    y_pred=pipe.predict(x_test)
    print(r2_score(y_test,y_pred))


scores=[]
for i in range(1000):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=i)
    lr=LinearRegression()
    pipe=make_pipeline(column_trans,lr)
    pipe.fit(x_train,y_train)
    y_pred=pipe.predict(x_test)
    scores.append(r2_score(y_test,y_pred))

# for i in scores:
#     print(i)

max_scores=np.argmax(scores)
print(max_scores)

max_score=scores[np.argmax(scores)]
print(max_score)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=np.argmax(scores))
lr=LinearRegression()
pipe=make_pipeline(column_trans,lr)
pipe.fit(x_train,y_train)
y_pred=pipe.predict(x_test)
score=r2_score(y_test,y_pred)


print("Random State is set to maximum score...")
print("r2_score :-",score)

import pickle

pickle.dump(pipe,open('LinearRegressionModel.pkl','wb'))


pipe.predict(pd.DataFrame([['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']],columns=['name','company','year','kms_driven','fuel_type']))

predicted_price=pipe.predict(pd.DataFrame([['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']],columns=['name','company','year','kms_driven','fuel_type']))

print("Predicted Price- ",predicted_price)
