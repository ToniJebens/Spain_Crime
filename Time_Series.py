# Predict Crime Rates for the next 3 years by location us Prophet

import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Data
data = pd.read_csv('crime_rate_Spain.csv')
df = pd.DataFrame(data)

# Convert 'Year' column to datetime
df['Year'] = pd.to_datetime(df['Year'], format='%Y')


forecasts = {}  # store forecasts for each location

# Loop over each unique location
for location in df['Location'].unique():

    # Filter dataframe by location
    loc_data = df[df['Location'] == location].copy()
    
    # change columns to 'ds' and 'y' (prophet requirements)
    loc_data.rename(columns={'Year': 'ds', 'Total cases': 'y'}, inplace=True)
    
    # fit data
    model = Prophet(yearly_seasonality=True)
    model.fit(loc_data)
    
    # df for future dates 
    future = model.make_future_dataframe(periods=3, freq='Y')
    
    # Predict
    forecast = model.predict(future)
    forecasts[location] = forecast
    
    # Plot
    fig = model.plot(forecast, xlabel='Year', ylabel='Total Cases')
    plt.title(f'Forecast of Total Cases for {location}')
    plt.show()

