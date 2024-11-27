import streamlit as st
import numpy as np
import pandas as pd
import pickle
import requests
from datetime import datetime

# Load the model
with open('house_price_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

def validate_pincode(zipcode):
    url = f"https://api.postalpincode.in/pincode/{zipcode}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data and 'PostOffice' in data[0] and data[0]['PostOffice']:
                area_info = data[0]['PostOffice'][0]
                state_name = area_info['State']
                area_name = area_info['Name']
                # Check for Tamil Nadu or Kerala
                if state_name.lower() in ["tamil nadu", "kerala"]:
                    return area_name, state_name
                else:
                    return None, None
        return None, None
    except requests.exceptions.RequestException:
        st.error("API request failed. Please try again later.")
        return None, None


# Title and description
st.title("Indian House Price Prediction")
st.write("This application predicts the price of a house in India based on several input features.")

# Input fields
sqft_living = st.number_input("Square Footage of Living Area (sqft)", min_value=100, value=1500)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
floors = st.number_input("Number of Floors", min_value=1, max_value=10, value=1)
age_of_property = st.number_input("Age of Property (years)", min_value=0, value=5)
pincode = st.number_input("Pincode", min_value=100000, max_value=999999, value=600001)

# New Date Input for Sold Date
sold_date = st.date_input("Sold Date", value=datetime.today())
year_sold = sold_date.year
month_sold = sold_date.month
quarter_sold = (month_sold - 1) // 3 + 1  # Calculate quarter

# Common values for the last five years
yoy_price_change = 5.50  # Year-on-Year Price Change (%)
demand_index = 102.00     # Demand Index
inflation_rate = 6.10     # Inflation Rate (%)
gdp_growth = 7.30         # GDP Growth (%)

# Calculate Season Sold based on the sold date
season_map = {
    1: "Summe",   # January
    2: "Summe",   # February
    3: "Summer",   # March
    4: "Summer",   # April
    5: "Summer",   # May
    6: "Summer",   # June
    7: "Summer",   # July
    8: "Summer",   # August
    9: "Summer",   # September
    10: "Summer",  # October
    11: "Summer",  # November
    12: "Summer"   # December
}
season = season_map.get(month_sold, "Unknown")  # Default to "Unknown" if not found

years_since_event = st.number_input("Years Since Last Event", min_value=0, value=2)
development_proximity = st.slider("Development Project Proximity (km)", min_value=0.0, max_value=10.0, value=1.0)
event_occurred = st.selectbox("Was there a recent major event in the area?", ["Yes", "No"])

# Calculate engineered features
sqft_age = sqft_living * age_of_property
price_per_bedroom = sqft_living / bedrooms if bedrooms != 0 else 0

# Validate pincode
if st.button("Predict Price"):
    area_name, state_name = validate_pincode(pincode)

    if state_name in ["Tamil Nadu", "Kerala"]:
        # Create DataFrame for prediction
        new_data = pd.DataFrame({
            'Sqft_Living': [sqft_living],
            'Bedrooms': [bedrooms],
            'Bathrooms': [bathrooms],
            'Floors': [floors],
            'Age_of_Property': [age_of_property],
            'Pincode': [pincode],
            'Year_Sold': [year_sold],
            'Month_Sold': [month_sold],
            'Quarter_Sold': [quarter_sold],
            'YoY_Price_Change_(%)': [yoy_price_change],
            'Demand_Index': [demand_index],
            'Inflation_Rate_(%)': [inflation_rate],
            'GDP_Growth_(%)': [gdp_growth],
            'Years_Since_Event': [years_since_event],
            'Development_Project_Proximity': [development_proximity],
            'Event_Occurred': [event_occurred],
            'Season': [season],
            'Sqft_Age': [sqft_age],
            'Price_per_Bedroom': [price_per_bedroom]
        })

        # Make prediction
        predicted_price_log = loaded_model.predict(new_data)
        predicted_price = np.expm1(predicted_price_log)  # Revert log transformation
        
        # Display results
        st.markdown(f"<strong>Area:</strong> {area_name}, <strong>State:</strong> {state_name}", unsafe_allow_html=True)
        st.markdown(f"<strong>Predicted House Price:</strong> ₹{predicted_price[0]:,.2f}", unsafe_allow_html=True)

    else:
        st.error("The provided pincode does not belong to Tamil Nadu or Kerala, or is invalid.")
