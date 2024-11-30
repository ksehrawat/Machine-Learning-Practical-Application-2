# Machine-Learning-Practical-Application-2
The main aim of  the project is to predict the price of used cars based on various attributes provided in the dataset. This involves identifying key features that influence car prices (e.g., make, model, year, mileage, and condition). The goal is to help the dealership better understand which factors increase a car's value and guide inventory and pricing strategies.

# Data Information

* Data Folder: **Data** - This folder is created to store the data file (vehicles.csv). This Folder also has another file filtered_vehicles.csv which is the copy of the clean data. Folder is saved as a Zip file as we cannot upload the whole folder due to the Size Limit of 25MB

* Data Set - Data set used in the project is in the **vehicles.csv** file 
  
* Code File - Python Code is stored in the **Machine_Learning_Practical_Application_2.ipynb** file

# Create DataFrame in Python for the Data Set

```python
# Python Code to  read the vechicles.csv stored in the Data Folder using various Pyhon Libraries

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
file_path = 'Data/vehicles.csv'
vehicles_df = pd.read_csv(file_path)


