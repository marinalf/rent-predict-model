# Python Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Data Set
data = pd.read_csv("nyc_apts.csv")

# Data Preparation
print(data.head())
print(data.isnull().sum())
print(data.describe())

print(f"Mean Rent: {data.Rent.mean()}")
print(f"Median Rent: {data.Rent.median()}")
print(f"Highest Rent: {data.Rent.max()}")
print(f"Lowest Rent: {data.Rent.min()}")

# Visualization
figure = px.bar(data, x=data["Borough"], 
                y = data["Rent"], 
                color = data["Rooms"],
            title="Rent in different boroughs of New York City according to number of bedrooms")
figure.show()

figure = px.bar(data, x=data["Borough"], 
                y = data["Rent"], 
                color = data["Size"],
            title="Rent in different boroughs according to size")
figure.show()


cities = data["Borough"].value_counts()
label = cities.index
counts = cities.values
colors = ['gold','lightgreen']

fig = go.Figure(data=[go.Pie(labels=label, values=counts, hole=0.5)])
fig.update_layout(title_text='Number of Houses Available for Rent')
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=30,
                  marker=dict(colors=colors, line=dict(color='black', width=3)))
fig.show()

# Converting categorical into numerical features
data["Area Type"] = data["Area Type"].map({"Super Area": 1, 
                                           "Carpet Area": 2, 
                                           "Built Area": 3})
data["Borough"] = data["Borough"].map({"Queens": 4000, "Brooklyn": 6000, 
                                 "Manhattan": 5600})
data["Furnishing Status"] = data["Furnishing Status"].map({"Unfurnished": 0,  
                                                           "Furnished": 1})
data["Tenant Preferred"] = data["Tenant Preferred"].map({"Bachelors/Family": 2, 
                                                         "Bachelors": 1, 
                                                         "Family": 3})

# Splitting data and test sets
from sklearn.model_selection import train_test_split

x = np.array(data[["Rooms", "Size", "Area Type", "Borough", 
                   "Furnishing Status", "Tenant Preferred"]])
y = np.array(data[["Rent"]])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.10, 
                                                random_state=42)

xtrain = xtrain.reshape((xtrain.shape[0], xtrain.shape[1], 1))
xtest = xtest.reshape((xtest.shape[0], xtest.shape[1], 1))

# Neural Network

from keras.models import Sequential # type: ignore
from keras.layers import Input, Dense, LSTM # type: ignore
from keras.optimizers import Adam # type: ignore

# Model Config

model = Sequential()
model.add(Input(shape=(xtrain.shape[1], xtrain.shape[2])))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25, activation='relu'))
model.add(Dense(1))

model.summary()

# Model Compile

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
model.fit(xtrain, ytrain, batch_size=32, epochs=21)

# Save the model
model.save('my_model.keras')  # Saves the model in the current directory

