import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Visualize image pixel values
plt.figure(figsize=(8, 6))
plt.boxplot(data.reshape(-1))
plt.title('Boxplot of Image Pixel Values')
plt.xlabel('Pixel Values')
plt.ylabel('Frequency')
plt.show()

# Calculate IQR for image pixel values
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1

# Identify outliers based on IQR
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Outlier detection
outliers = (data < lower_bound) | (data > upper_bound)

# Handling outliers by capping them
data[outliers] = np.clip(data[outliers], lower_bound, upper_bound)

# Data Type Conversion
# Convert class labels to categorical type
labels = pd.Categorical(labels)

# Dimensionality
print("Data shape:", data.shape)  # Print the shape of the data array

# Descriptive Statistics
# Convert data array to DataFrame for descriptive statistics
data_df = pd.DataFrame(data.reshape(data.shape[0], -1))

# Summary statistics for numerical features
summary_stats = data_df.describe()

print("\nSummary Statistics for Numerical Features:")
print(summary_stats)

