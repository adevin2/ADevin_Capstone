**Product Count Prediction for OTC Drugs
***Project Description***
Purpose: This is an app in which the user can input 1) a year and 2) product type: liquid, tablet, and cream. The app will output a predicted number of OTC Drug Products in that product type category in that year. 

Potential Audience: Pharmaceutical companies, market analysts, distributors, startups for OTC products, students

Usage:
1) Install python 3.7+
2) Install streamlit (command: pip install streamlit)
3) Clone the repository for "Capstone.py"
4) Install all dependencies:
    pandas
    numpy
    streamlit
    sklearn.model_selection
    sklearn.linear_model
    plotly.express
    plotly.graph_objects
5) In terminal run "streamlit run Capstone.py"
6) Use the dropdown to select the product type of interest.
7) Use the slider to select a year.

OUTPUT: A number for the predicted number of products for that year based on the linear regression model. An interactive plot will also be present to view the prediction compared to the actual data. 

***Version***
v1.0.0

***Author***
Alese Devin

***Project Contents Breakdown***
The following repo contains all the contents for the project: https://github.com/adevin2/ADevin_Capstone.git
- Initial Dataset - https://www.kaggle.com/datasets/maheshdadhich/us-healthcare-data/data?select=Drugs_product.csv
- ADevin_Capstone.ipynb - This document breaks down the data cleanup in detail, then breaksdown the preliminary thought process for the app and explores different avenues for the app. 
- Capstone.py - the code for the app itself.
- Drug_product.csv and Drugs_product.txt - input datasets used for this app development.
- Video Link: https://youtu.be/eYz28rkb9fM
