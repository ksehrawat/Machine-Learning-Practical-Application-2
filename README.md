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

# Print the column names and their data types
print(vehicles_df.info())

#   Column        Non-Null Count   Dtype  
---  ------        --------------   -----  
 0   id            426880 non-null  int64  
 1   region        426880 non-null  object 
 2   price         426880 non-null  int64  
 3   year          425675 non-null  float64
 4   manufacturer  409234 non-null  object 
 5   model         421603 non-null  object 
 6   condition     252776 non-null  object 
 7   cylinders     249202 non-null  object 
 8   fuel          423867 non-null  object 
 9   odometer      422480 non-null  float64
 10  title_status  418638 non-null  object 
 11  transmission  424324 non-null  object 
 12  VIN           265838 non-null  object 
 13  drive         296313 non-null  object 
 14  size          120519 non-null  object 
 15  type          334022 non-null  object 
 16  paint_color   296677 non-null  object 
 17  state         426880 non-null  object 

# Print the data frame first 5 rows in a structured format

print(vehicles_df.head().to_markdown(index=False, numalign="left", stralign="left"))

| id         | region                 | price   | year   | manufacturer   | model   | condition   | cylinders   | fuel   | odometer   | title_status   | transmission   | VIN   | drive   | size   | type   | paint_color   | state   |
|:-----------|:-----------------------|:--------|:-------|:---------------|:--------|:------------|:------------|:-------|:-----------|:---------------|:---------------|:------|:--------|:-------|:-------|:--------------|:--------|
| 7222695916 | prescott               | 6000    | nan    | nan            | nan     | nan         | nan         | nan    | nan        | nan            | nan            | nan   | nan     | nan    | nan    | nan           | az      |
| 7218891961 | fayetteville           | 11900   | nan    | nan            | nan     | nan         | nan         | nan    | nan        | nan            | nan            | nan   | nan     | nan    | nan    | nan           | ar      |
| 7221797935 | florida keys           | 21000   | nan    | nan            | nan     | nan         | nan         | nan    | nan        | nan            | nan            | nan   | nan     | nan    | nan    | nan           | fl      |
| 7222270760 | worcester / central MA | 1500    | nan    | nan            | nan     | nan         | nan         | nan    | nan        | nan            | nan            | nan   | nan     | nan    | nan    | nan           | ma      |
| 7210384030 | greensboro             | 4900    | nan    | nan            | nan     | nan         | nan         | nan    | nan        | nan            | nan            | nan   | nan     | nan    | nan    | nan           | nc      |
```
