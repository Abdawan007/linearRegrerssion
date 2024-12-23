import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Title for the app
st.title("Linear Regression with Streamlit")

# Instructions
st.write("""
This application allows you to input data points (X and Y values), train a Linear Regression model, and visualize the results.
You can see the regression line and the model's performance metrics.
""")

# Sidebar for input
st.sidebar.header("Input Data")

# Number of data points user wants to input
num_points = st.sidebar.slider("Select number of data points", min_value=10, max_value=100, value=30)

# Empty lists to store X and Y data
x_data = []
y_data = []

# Collect input data from the user
for i in range(num_points):
    x = st.sidebar.number_input(f"Enter X value for data point {i + 1}", key=f"x_{i}", step=0.1)
    y = st.sidebar.number_input(f"Enter Y value for data point {i + 1}", key=f"y_{i}", step=0.1)
    x_data.append(x)
    y_data.append(y)

# Convert input data into numpy arrays for training
X = np.array(x_data).reshape(-1, 1)  # Reshaping X for scikit-learn
Y = np.array(y_data)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create the Linear Regression model
model = LinearRegression()

# Train the model with the training data
model.fit(X_train, Y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Display Model's Evaluation Metrics
st.subheader("Model Evaluation:")
st.write(f"**Intercept**: {model.intercept_}")
st.write(f"**Coefficient (Slope)**: {model.coef_[0]}")

# Calculate and display Mean Squared Error (MSE) and R² Score
mse = mean_squared_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)

st.write(f"**Mean Squared Error (MSE)**: {mse}")
st.write(f"**R² Score**: {r2}")

# Plotting the original data points and regression line
st.subheader("Regression Line Plot")

# Scatter plot for the original data points
plt.scatter(X, Y, color="blue", label="Data Points")

# Plot the regression line
plt.plot(X, model.predict(X), color="red", label="Regression Line")

# Labels and legend
plt.xlabel("X (Independent Variable)")
plt.ylabel("Y (Dependent Variable)")
plt.legend()

# Display the plot in the Streamlit app
st.pyplot(plt)
