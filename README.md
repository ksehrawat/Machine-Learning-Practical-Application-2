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
# Data Analysis and Cleaning

After our initial exploration and fine-tuning of the business understanding, it is time to construct our final dataset prior to modeling. Here, we want to make sure to handle any integrity issues and cleaning, the engineering of new features, any transformations that we believe should happen

```python
# Print descriptive statistics for all numeric columns

print("\nDescriptive Statistics for Numeric Columns:\n")
print(vehicles_df.describe().to_markdown(numalign="left", stralign="left"))

# For all object type columns, print the number of distinct values and the most frequent value

print("\nObject Column Summaries:\n")
for col in vehicles_df.select_dtypes(include='object'):
    print(f"Column: {col}")
    print(f"  Number of distinct values: {vehicles_df[col].nunique()}")
    print(f"  Most frequent value: {vehicles_df[col].mode()[0]}\n")

Object Column Summaries:

Column: region
  Number of distinct values: 404
  Most frequent value: columbus

Column: manufacturer
  Number of distinct values: 42
  Most frequent value: ford

Column: model
  Number of distinct values: 29649
  Most frequent value: f-150

Column: condition
  Number of distinct values: 6
  Most frequent value: good

Column: cylinders
  Number of distinct values: 8
  Most frequent value: 6 cylinders

Column: fuel
  Number of distinct values: 5
  Most frequent value: gas

Column: title_status
  Number of distinct values: 6
  Most frequent value: clean

Column: transmission
  Number of distinct values: 3
  Most frequent value: automatic

Column: VIN
  Number of distinct values: 118246
  Most frequent value: 1FMJU1JT1HEA52352

Column: drive
  Number of distinct values: 3
  Most frequent value: 4wd

Column: size
  Number of distinct values: 4
  Most frequent value: full-size

Column: type
  Number of distinct values: 13
  Most frequent value: sedan

Column: paint_color
  Number of distinct values: 12
  Most frequent value: white

Column: state
  Number of distinct values: 51
  Most frequent value: ca


# Print the count and percentage of missing values for each column
missing_values = vehicles_df.isnull().sum()
missing_percent = (missing_values / len(vehicles_df)) * 100
print("Missing Values:\n")
print(pd.concat([missing_values, missing_percent], axis=1, keys=['Count', 'Percentage']).sort_values(by='Count', ascending=False).to_markdown(numalign="left", stralign="left"))

Missing Values:

|              | Count   | Percentage   |
|:-------------|:--------|:-------------|
| size         | 306361  | 71.7675      |
| cylinders    | 177678  | 41.6225      |
| condition    | 174104  | 40.7852      |
| VIN          | 161042  | 37.7254      |
| drive        | 130567  | 30.5863      |
| paint_color  | 130203  | 30.5011      |
| type         | 92858   | 21.7527      |
| manufacturer | 17646   | 4.13371      |
| title_status | 8242    | 1.93075      |
| model        | 5277    | 1.23618      |
| odometer     | 4400    | 1.03073      |
| fuel         | 3013    | 0.705819     |
| transmission | 2556    | 0.598763     |
| year         | 1205    | 0.282281     |
| id           | 0       | 0            |
| region       | 0       | 0            |
| price        | 0       | 0            |
| state        | 0       | 0            |

# Check for duplicate VIN numbers and count them
duplicate_vin_count = vehicles_df['VIN'].duplicated(keep=False).sum()
print(duplicate_vin_count)

348914

# Drop specified columns from the vehicles_df dataframe
vehicles_df.drop(columns=['size', 'cylinders', 'condition', 'VIN', 'drive', 'paint_color'], inplace=True)

# Display the remaining columns to confirm the drop
vehicles_df.columns

Index(['id', 'region', 'price', 'year', 'manufacturer', 'model', 'fuel',
       'odometer', 'title_status', 'transmission', 'type', 'state'],
      dtype='object')

# Drop all rows with any null values from the remaining columns
vehicles_df.dropna(inplace=True)

# Check the dataset info to confirm removal of null values
vehicles_df.info()

Data columns (total 12 columns):
 #   Column        Non-Null Count   Dtype  
---  ------        --------------   -----  
 0   id            306976 non-null  int64  
 1   region        306976 non-null  object 
 2   price         306976 non-null  int64  
 3   year          306976 non-null  float64
 4   manufacturer  306976 non-null  object 
 5   model         306976 non-null  object 
 6   fuel          306976 non-null  object 
 7   odometer      306976 non-null  float64
 8   title_status  306976 non-null  object 
 9   transmission  306976 non-null  object 
 10  type          306976 non-null  object 
 11  state         306976 non-null  object

All rows with null values have been successfully removed, leaving us with 306,976 entries and 12 columns. The dataset now contains no missing values in any of the remaining columns. ​

Remove any outliers from the numerical columns (price, year, or odometer) of the dataset.

The steps will include:

Define Outlier Boundaries: Using the Interquartile Range (IQR) method, we can set boundaries for each column.

Filter Outliers: Remove rows where price, year, or odometer values fall outside the defined boundaries.

# Define function to remove outliers based on the IQR method
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers in 'price', 'year', and 'odometer' columns
filtered_df = remove_outliers(vehicles_df, 'price')
filtered_df = remove_outliers(filtered_df, 'year')
filtered_df = remove_outliers(filtered_df, 'odometer')

# Display the updated dataset info to confirm
filtered_df.info()

Data columns (total 12 columns):
 #   Column        Non-Null Count   Dtype  
---  ------        --------------   -----  
 0   id            291256 non-null  int64  
 1   region        291256 non-null  object 
 2   price         291256 non-null  int64  
 3   year          291256 non-null  float64
 4   manufacturer  291256 non-null  object 
 5   model         291256 non-null  object 
 6   fuel          291256 non-null  object 
 7   odometer      291256 non-null  float64
 8   title_status  291256 non-null  object 
 9   transmission  291256 non-null  object 
 10  type          291256 non-null  object 
 11  state         291256 non-null  object

# Remove rows where the price is less than 500
filtered_df = filtered_df[filtered_df['price'] >= 500]

Outliers have been removed from the price, year, and odometer columns, resulting in a dataset with 291,256 entries. This cleaning process should improve model accuracy by reducing the influence of extreme values.

# Convert the 'id' field to a non-numeric value by changing it to a string data type
filtered_df['id'] = filtered_df['id'].astype(str)

# Confirm the change by checking the data type of 'id' column
print(filtered_df['id'].dtype)

# Convert specified columns to uppercase
columns_to_upper = ['model', 'region', 'manufacturer', 'fuel', 'title_status', 'transmission', 'type', 'state']
filtered_df[columns_to_upper] = filtered_df[columns_to_upper].apply(lambda x: x.str.upper())

# Confirm by displaying the first few rows of the modified columns
print(filtered_df[columns_to_upper].head())

# Rename columns to have the first letter as uppercase
filtered_df.columns = [col.capitalize() for col in filtered_df.columns]

# Confirm by displaying the renamed columns
filtered_df.columns

 Rename 'Title_status' column to 'Title Status'
filtered_df.rename(columns={'Title_status': 'Title Status'}, inplace=True)

# Confirm by displaying the columns
print(filtered_df.columns)

# Save the clean data file in the Data Folder for future reference

filtered_file_path = 'Data/filtered_vehicles.csv'
filtered_df.to_csv(filtered_file_path, index=False)
