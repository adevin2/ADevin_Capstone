import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go

#Raw file url for the upload
url = 'https://raw.githubusercontent.com/adevin2/ADevin_Capstone/main/Drugs_product.csv'

#Load the dataset with encoding
df = pd.read_csv(url, encoding='ISO-8859-1')

#Data cleaning
df_cleaned = df.drop(columns=['DEASCHEDULE', 'PHARM_CLASSES', 'ACTIVE_INGRED_UNIT', 
                              'ACTIVE_NUMERATOR_STRENGTH', 'PROPRIETARYNAMESUFFIX', 'APPLICATIONNUMBER', 'ENDMARKETINGDATE', 'LABELERNAME'])
df_filtered = df_cleaned[df_cleaned['PRODUCTTYPENAME'] == 'HUMAN OTC DRUG']
df_filtered = df_filtered.dropna(subset=['STARTMARKETINGDATE'])
df_filtered['STARTMARKETINGDATE'] = pd.to_datetime(df_filtered['STARTMARKETINGDATE'].astype(int).astype(str), format='%Y%m%d', errors='coerce')
df_filtered['DOSAGEFORMNAME'] = df_filtered['DOSAGEFORMNAME'].str.split('[,;]').str[0]
df_filtered['ROUTENAME'] = df_filtered['ROUTENAME'].str.split('[,;]').str[0]
df_filtered = df_filtered[df_filtered['DOSAGEFORMNAME'].isin(['LIQUID', 'TABLET', 'CREAM'])]
df_filtered['Year'] = df_filtered['STARTMARKETINGDATE'].dt.year

#Group data by Year and DOSAGEFORMNAME, counting the number of products per year for each type
product_counts = df_filtered.groupby(['Year', 'DOSAGEFORMNAME']).size().reset_index(name='Product_Count')

#Split the data for each product type
liquid_df = product_counts[product_counts['DOSAGEFORMNAME'] == 'LIQUID']
tablet_df = product_counts[product_counts['DOSAGEFORMNAME'] == 'TABLET']
cream_df = product_counts[product_counts['DOSAGEFORMNAME'] == 'CREAM']

#Log-transform the Product_Count
liquid_df['Log_Product_Count'] = np.log(liquid_df['Product_Count'])
tablet_df['Log_Product_Count'] = np.log(tablet_df['Product_Count'])
cream_df['Log_Product_Count'] = np.log(cream_df['Product_Count'])

#Function to train and predict for each product type with log-transformed values
def train_predict_log_model(df):
    X = df[['Year']]  # Independent variable
    y = df['Log_Product_Count']  # Dependent variable
    
    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Reverse log transform to make it interpretable
    y_pred_original_scale = np.expm1(y_pred)
    
    return model, X_test, y_test, y_pred_original_scale

# Train and predict for LIQUID, TABLET, and CREAM with log transformation
liquid_model, X_test_liquid, y_test_liquid, y_pred_liquid = train_predict_log_model(liquid_df)
tablet_model, X_test_tablet, y_test_tablet, y_pred_tablet = train_predict_log_model(tablet_df)
cream_model, X_test_cream, y_test_cream, y_pred_cream = train_predict_log_model(cream_df)

#Streamlit app setup
st.title('Product Count Prediction for OTC Drugs')

#Dropdown for product type
product_type = st.selectbox('Select Product Type', ['LIQUID', 'TABLET', 'CREAM'])

#User input slider for year prediction
year = st.slider('Select Year for Prediction', 1980, 2050, 2020)

#Predict based on selection and input year
if product_type == 'LIQUID':
    prediction = liquid_model.predict([[year]])  # Predict for the selected year
    predicted_count = np.expm1(prediction)[0]  # Reverse log transform
    st.write(f"Predicted Product Count for LIQUID in {year}: {predicted_count:.0f}")
    historical_data = liquid_df
    predicted_year_data = pd.DataFrame({'Year': [year], 'Product_Count': [predicted_count]})

elif product_type == 'TABLET':
    prediction = tablet_model.predict([[year]])  # Predict for the selected year
    predicted_count = np.expm1(prediction)[0]  # Reverse log transform
    st.write(f"Predicted Product Count for TABLET in {year}: {predicted_count:.0f}")
    historical_data = tablet_df
    predicted_year_data = pd.DataFrame({'Year': [year], 'Product_Count': [predicted_count]})

else:  # For CREAM
    prediction = cream_model.predict([[year]])  # Predict for the selected year
    predicted_count = np.expm1(prediction)[0]  # Reverse log transform
    st.write(f"Predicted Product Count for CREAM in {year}: {predicted_count:.0f}")
    historical_data = cream_df
    predicted_year_data = pd.DataFrame({'Year': [year], 'Product_Count': [predicted_count]})

#Combine historical data and predicted data
combined_data = pd.concat([historical_data, predicted_year_data])

#Create the scatter plot
fig = go.Figure()

#Add historical data to the plot
fig.add_trace(go.Scatter(x=historical_data['Year'], y=historical_data['Product_Count'],
                         mode='markers', name='Historical Data', marker=dict(color='blue')))

#Add predicted data to the plot
fig.add_trace(go.Scatter(x=predicted_year_data['Year'], y=predicted_year_data['Product_Count'],
                         mode='markers', name='Predicted Data', marker=dict(color='red', size=12)))

#Update layout
fig.update_layout(
    title=f"Historical vs Predicted Product Count for {product_type}",
    xaxis_title="Year",
    yaxis_title="Product Count",
    showlegend=True
)

#Display the plot
st.plotly_chart(fig)

#Run the app
if __name__ == '__main__':
    st.write("""
    This app predicts the number of over-the-counter (OTC) drug products for a given year based on historical data.
    The model was trained using a linear regression model with log-transformed product counts.
    """)
