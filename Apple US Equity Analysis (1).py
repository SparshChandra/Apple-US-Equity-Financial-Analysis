#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd

# File path to the dataset
file_path = 'Apple US Equity Data.xlsx'

# Read the Excel file, skipping the first row and using the second row as the header
apple_data = pd.read_excel(file_path, header=1)

# Display the first few rows to confirm
print(apple_data.head())


# In[25]:


print("Original column names:", apple_data.columns.tolist())

# Remove rows that are completely empty
apple_data.dropna(how='all', inplace=True)

# Rename columns for clarity (modify as per actual column names in your dataset)
column_names = {
    'PX_HIGH': 'High Price', 
    'PX_LOW': 'Low Price', 
    'PX_VOLUME': 'Volume', 
    'PX_LAST': 'Last Price', 
    'PX_MID': 'Mid Price'
}
apple_data.rename(columns=column_names, inplace=True)

# Check if the renaming was successful
print("Renamed column names:", apple_data.columns.tolist())

# Proceed with the rest of the cleaning process

# Convert 'Volume' to integers (make sure the column name matches exactly)
# First, check if 'Volume' column exists to avoid KeyError
if 'Volume' in apple_data.columns:
    apple_data['Volume'] = pd.to_numeric(apple_data['Volume'], errors='coerce').fillna(0).astype(int)
else:
    print("'Volume' column not found.")

# Handle missing values for other columns as needed
# Example: Filling missing values in 'High Price' with the mean
# Make sure to check if the column exists
if 'High Price' in apple_data.columns:
    apple_data['High Price'].fillna(apple_data['High Price'].mean(), inplace=True)
else:
    print("'High Price' column not found.")

# Display the cleaned data
print(apple_data.head())


# In[26]:


# Manually specify column names based on your data structure
column_names = ['High Price', 'Low Price', 'Volume', 'Last Price', 'Mid Price']

# Read the Excel file, skipping rows that do not contain the actual data
apple_data = pd.read_excel(file_path, skiprows=1, header=None, names=column_names)

# Convert columns to numeric, setting errors='coerce' to turn non-convertible values into NaN
for col in ['High Price', 'Low Price', 'Volume', 'Last Price', 'Mid Price']:
    apple_data[col] = pd.to_numeric(apple_data[col], errors='coerce')

# Replace NaN values with the mean in the 'High Price' column
apple_data['High Price'].fillna(apple_data['High Price'].mean(), inplace=True)

# Optionally, handle NaN values in other columns
# For example, replacing NaN in 'Volume' with 0
apple_data['Volume'].fillna(0, inplace=True)

# Convert 'Volume' to integer
apple_data['Volume'] = apple_data['Volume'].astype(int)

# Display the cleaned data
print(apple_data.head())


# In[28]:



# Convert columns to numeric
for col in column_names:
    apple_data[col] = pd.to_numeric(apple_data[col], errors='coerce')

# Fill missing values
apple_data.fillna(method='ffill', inplace=True)  # Forward fill for time series data

# Descriptive Statistics
descriptive_stats = apple_data.describe()

# Calculate a simple moving average (SMA) for the 'Last Price' - 30 days window
apple_data['30 Day SMA'] = apple_data['Last Price'].rolling(window=30).mean()

# Print Descriptive Statistics
print("Descriptive Statistics:\n", descriptive_stats)

# Print the head of the dataframe to show the SMA column
print("\nData with 30 Day SMA:\n", apple_data.head())

# Further analysis can be done based on specific financial analysis requirements


# In[29]:


# Convert columns to numeric and handle missing values
for col in column_names:  # All columns in the list
    apple_data[col] = pd.to_numeric(apple_data[col], errors='coerce')
    apple_data[col].fillna(method='ffill', inplace=True)  # Forward fill for time series data

# Feature Engineering
# Calculate daily price change (only if the dataset is ordered by date)
if 'Last Price' in apple_data.columns:
    apple_data['Daily Change'] = apple_data['Last Price'].diff()

# Calculate moving averages (e.g., 7-day and 30-day), assuming data is ordered by date
if 'Last Price' in apple_data.columns:
    apple_data['7 Day SMA'] = apple_data['Last Price'].rolling(window=7).mean()
    apple_data['30 Day SMA'] = apple_data['Last Price'].rolling(window=30).mean()

# Display the processed data
print(apple_data.head())


# In[30]:


# Convert columns to numeric
for col in column_names:
    apple_data[col] = pd.to_numeric(apple_data[col], errors='coerce')

# Drop rows where all elements are NaN
apple_data.dropna(how='all', inplace=True)

# Forward fill the remaining NaN values
apple_data.fillna(method='ffill', inplace=True)

# Feature Engineering
apple_data['Daily Change'] = apple_data['Last Price'].diff()
apple_data['7 Day SMA'] = apple_data['Last Price'].rolling(window=7).mean()
apple_data['30 Day SMA'] = apple_data['Last Price'].rolling(window=30).mean()

# Display the processed data
print(apple_data.head())


# In[31]:


import matplotlib.pyplot as plt

# Convert 'High Price' to numeric, just in case it's not already
apple_data['High Price'] = pd.to_numeric(apple_data['High Price'], errors='coerce')

# Drop rows where 'High Price' is NaN which could result from conversion errors
apple_data.dropna(subset=['High Price'], inplace=True)

# Reset index to ensure it's sequential after dropping NaNs
apple_data.reset_index(drop=True, inplace=True)

# Plotting
plt.figure(figsize=(14, 7))  # Set the size of the plot
plt.plot(apple_data.index, apple_data['High Price'], label='High Price')  # Plot the 'High Price' column
plt.title('High Price Trend')  # Title of the plot
plt.xlabel('Time (Index)')  # X-axis label
plt.ylabel('High Price')  # Y-axis label
plt.legend()  # Show legend
plt.grid(True)  # Show grid
plt.show()  # Display the plot


# In[35]:


# Convert 'Low Price' to numeric, just in case it's not already
apple_data['Low Price'] = pd.to_numeric(apple_data['Low Price'], errors='coerce')

# Drop rows where 'Low Price' is NaN which could result from conversion errors
apple_data.dropna(subset=['Low Price'], inplace=True)

# Reset index to ensure it's sequential after dropping NaNs
apple_data.reset_index(drop=True, inplace=True)

# Plotting
plt.figure(figsize=(14, 7))  # Set the size of the plot
plt.scatter(apple_data.index, apple_data['Low Price'], label='Low Price')  # Plot the 'Low Price' column
plt.title('Low Price Trend')  # Title of the plot
plt.xlabel('Time (Index)')  # X-axis label
plt.ylabel('Low Price')  # Y-axis label
plt.legend()  # Show legend
plt.grid(True)  # Show grid
plt.show()  # Display the plot


# In[38]:


# Convert 'Last Price' to numeric, just in case it's not already
apple_data['Last Price'] = pd.to_numeric(apple_data['Last Price'], errors='coerce')

# Drop rows where 'Low Price' is NaN which could result from conversion errors
apple_data.dropna(subset=['Last Price'], inplace=True)

# Reset index to ensure it's sequential after dropping NaNs
apple_data.reset_index(drop=True, inplace=True)

# Plotting
plt.figure(figsize=(14, 7))  # Set the size of the plot
plt.plot(apple_data.index, apple_data['Last Price'], label='Last Price')  # Plot the 'Last Price' column
plt.title('Last Price Trend')  # Title of the plot
plt.xlabel('Time (Index)')  # X-axis label
plt.ylabel('Low Price')  # Y-axis label
plt.legend()  # Show legend
plt.grid(True)  # Show grid
plt.show()  # Display the plot


# In[40]:


# Convert 'Mid Price' to numeric, just in case it's not already
apple_data['Mid Price'] = pd.to_numeric(apple_data['Mid Price'], errors='coerce')

# Drop rows where 'Mid Price' is NaN which could result from conversion errors
apple_data.dropna(subset=['Mid Price'], inplace=True)

# Reset index to ensure it's sequential after dropping NaNs
apple_data.reset_index(drop=True, inplace=True)

# Plotting
plt.figure(figsize=(14, 7))  # Set the size of the plot
plt.scatter(apple_data.index, apple_data['Mid Price'], label='Mid Price')  # Plot the 'Last Price' column
plt.title('Mid Price Trend')  # Title of the plot
plt.xlabel('Time (Index)')  # X-axis label
plt.ylabel('Mid Price')  # Y-axis label
plt.legend()  # Show legend
plt.grid(True)  # Show grid
plt.show()  # Display the plot


# In[42]:


# Convert 'Volume' to numeric, just in case it's not already
apple_data['Volume'] = pd.to_numeric(apple_data['Volume'], errors='coerce')

# Drop rows where 'Volume' is NaN which could result from conversion errors
apple_data.dropna(subset=['Volume'], inplace=True)

# Reset index to ensure it's sequential after dropping NaNs
apple_data.reset_index(drop=True, inplace=True)

# Plotting
plt.figure(figsize=(14, 7))  # Set the size of the plot
plt.plot(apple_data.index, apple_data['Volume'], label='Volume')  # Plot the 'Last Price' column
plt.title('Volume Trend')  # Title of the plot
plt.xlabel('Time (Index)')  # X-axis label
plt.ylabel('Volume')  # Y-axis label
plt.legend()  # Show legend
plt.grid(True)  # Show grid
plt.show()  # Display the plot


# In[45]:


# Convert all columns to numeric, just in case they're not already
columns_to_convert = ['High Price', 'Low Price', 'Mid Price', 'Last Price', 'Volume']
for col in columns_to_convert:
    apple_data[col] = pd.to_numeric(apple_data[col], errors='coerce')

# Drop rows with NaN values in any of the columns
apple_data.dropna(subset=columns_to_convert, inplace=True)

# Reset index to ensure it's sequential after dropping NaNs
apple_data.reset_index(drop=True, inplace=True)

# Plotting
plt.figure(figsize=(14, 7))

# Plotting Price Data
plt.plot(apple_data.index, apple_data['High Price'], label='High Price')
plt.plot(apple_data.index, apple_data['Low Price'], label='Low Price')
plt.plot(apple_data.index, apple_data['Mid Price'], label='Mid Price')
plt.plot(apple_data.index, apple_data['Last Price'], label='Last Price')

# Setting up the primary y-axis
plt.ylabel('Price')
plt.title('Comparison of High, Low, Mid, Last Prices, and Volume')
plt.legend(loc='upper left')

# Creating a secondary y-axis for Volume
ax2 = plt.twinx()
ax2.plot(apple_data.index, apple_data['Volume'], label='Volume', color='grey', linestyle='--')
ax2.set_ylabel('Volume')
ax2.legend(loc='upper right')

# Common X-axis label
plt.xlabel('Time (Index)')

plt.grid(True)
plt.show()


# In[50]:


import plotly.express as px
import pandas as pd

# Convert all columns to numeric, just in case they're not already
columns_to_convert = ['High Price', 'Low Price', 'Mid Price', 'Last Price', 'Volume']
for col in columns_to_convert:
    apple_data[col] = pd.to_numeric(apple_data[col], errors='coerce')

# Drop rows with NaN values in any of the columns
apple_data.dropna(subset=columns_to_convert, inplace=True)

# Resetting the index to use it as a time sequence
apple_data = apple_data.reset_index()

# Melt the DataFrame to make it suitable for Plotly Express
apple_data_melted = apple_data.melt(id_vars=['index'], value_vars=columns_to_convert, 
                                    var_name='Type', value_name='Value')

# Create an interactive line plot
fig = px.line(apple_data_melted, x='index', y='Value', color='Type',
              title='Interactive Comparison of High, Low, Mid, Last Prices, and Volume')

# Show the plot
fig.show()


# In[52]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Load a sample dataset
data = load_iris()
X, y = data.data, data.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),  # Increased max_iter for convergence
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier()
}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy * 100:.2f}%")


# In[53]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Load and preprocess your data
# apple_data = ...

# Define your target variable (y) and features (X)
# Example: Predicting if 'Last Price' will increase (1) or decrease (0) compared to the previous day
apple_data['Target'] = (apple_data['Last Price'].diff() > 0).astype(int)

X = apple_data[['High Price', 'Low Price', 'Volume', 'Mid Price']]  # Example feature set
y = apple_data['Target']

# Handle any NaN values in your feature set and target variable
X.fillna(method='ffill', inplace=True)
y.fillna(method='ffill', inplace=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier()
}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy * 100:.2f}%")


# In[54]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Assuming you have already split your data into training and test sets and trained your model
# Example:
# decision_tree_model = DecisionTreeClassifier()
# decision_tree_model.fit(X_train, y_train)
# y_pred = decision_tree_model.predict(X_test)

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix using Matplotlib
fig, ax = plt.subplots(figsize=(8, 8))
ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax)
plt.title("Confusion Matrix for Decision Tree")
plt.show()


# In[55]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

# decision_tree_model = DecisionTreeClassifier()
# decision_tree_model.fit(X_train, y_train)
# y_pred = decision_tree_model.predict(X_test)

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Creating the heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='g')
plt.title('Confusion Matrix Heatmap')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[60]:


import pandas as pd

# Assuming apple_data is your DataFrame

# Function to calculate RSI
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate MACD
def calculate_macd(series, short_period=12, long_period=26, signal_period=9):
    short_ema = series.ewm(span=short_period, adjust=False).mean()
    long_ema = series.ewm(span=long_period, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(series, window=20, num_std=2):
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

# Apply the functions to create the technical indicators
apple_data['RSI'] = calculate_rsi(apple_data['Last Price'])
apple_data['MACD'], apple_data['MACD_Signal'] = calculate_macd(apple_data['Last Price'])
apple_data['Upper_Band'], apple_data['Lower_Band'] = calculate_bollinger_bands(apple_data['Last Price'])

# Create lag features
for lag in [1, 5, 10]:
    apple_data[f'Last_Price_Lag_{lag}'] = apple_data['Last Price'].shift(lag)

# Create rolling statistics
apple_data['Rolling_Mean'] = apple_data['Last Price'].rolling(window=20).mean()
apple_data['Rolling_Std'] = apple_data['Last Price'].rolling(window=20).std()

# Handle NaN values
apple_data.dropna(inplace=True)

# Now you can proceed with your analysis or modeling with these new features.


# In[61]:


import matplotlib.pyplot as plt

# Visualize some of the features
plt.figure(figsize=(14, 7))
plt.plot(apple_data['Last Price'], label='Last Price')
plt.plot(apple_data['Rolling_Mean'], label='Rolling Mean')
plt.plot(apple_data['Upper_Band'], label='Upper Bollinger Band')
plt.plot(apple_data['Lower_Band'], label='Lower Bollinger Band')
plt.title('Stock Prices with Bollinger Bands')
plt.legend()
plt.show()


# In[62]:


# Importing necessary libraries
import matplotlib.pyplot as plt

# Summary and Recommendations for Apple Stock Price Analysis
def summary_and_recommendations(apple_data):
    """
    Summarizes the analysis performed on the Apple stock price dataset and provides recommendations for further steps.
    """

    # Summarize the Analysis Performed
    print("Summary of Analysis Performed:")
    print("1. Data Loading and Cleaning: Loaded historical Apple stock price data and performed necessary cleaning steps.")
    print("2. Feature Engineering: Added technical indicators (RSI, MACD, Bollinger Bands), lag features, and rolling statistics.")
    print("3. Exploratory Data Analysis: Conducted a preliminary analysis to understand the data characteristics and trends.")
    print("4. Advanced Feature Engineering: Calculated advanced financial indicators to capture market trends and stock price movements.")
    print("5. Visualization: Plotted key features and indicators for better understanding and insights.")
    print("6. Predictive Modeling: Suggested models for future stock price prediction or trend analysis.")
    
    # Recommendations for Further Steps
    print("\nRecommendations for Further Steps:")
    print("A. Model Development: Develop predictive models using the engineered features to forecast stock prices or classify market trends.")
    print("B. Model Evaluation: Evaluate model performance using appropriate metrics and cross-validation.")
    print("C. Hyperparameter Tuning: Optimize model parameters for better accuracy and performance.")
    print("D. Backtesting: Perform backtesting on historical data to assess the effectiveness of the predictive models.")
    print("E. Continuous Monitoring: Regularly update and monitor the model to adapt to new market conditions.")
    print("F. Deployment Strategy: If applicable, develop a strategy for deploying the model in a real-time trading environment.")

    # Optional: Display a sample visualization from the analysis
    plt.figure(figsize=(10, 5))
    apple_data['Last Price'].plot(title='Apple Stock Last Price Trend')
    plt.ylabel('Price')
    plt.show()

# Assuming apple_data is your pre-processed and feature-engineered DataFrame
summary_and_recommendations(apple_data)


# In[ ]:




