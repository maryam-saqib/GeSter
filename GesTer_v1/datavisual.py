import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Reshape the data array to flatten the images
num_samples, num_features = data.shape
data_flat = data.reshape(num_samples, -1)

# Convert data array to DataFrame for visualization
columns = [f'Pixel_{i}' for i in range(num_features)]
data_df = pd.DataFrame(data_flat, columns=columns)

# Numerical Features Visualization
plt.figure(figsize=(10, 6))
plt.boxplot(data.reshape(-1))
plt.title('Boxplot of Image Pixel Values')
plt.xlabel('Pixel Values')
plt.ylabel('Frequency')
plt.show()

# Categorical Features Visualization
plt.figure(figsize=(8, 6))
sns.countplot(labels)
plt.title('Distribution of Hand Gestures')
plt.xlabel('Frequency')
plt.ylabel('Hand Gesture')
plt.xticks(rotation=45)
plt.show()

# Scatter Plot (for demonstration purposes, considering only two pixels)
plt.figure(figsize=(8, 6))
sample_data = data_df.sample(n=1000, random_state=42)  # Sample 1000 rows for better visualization
plt.scatter(sample_data['Pixel_0'], sample_data['Pixel_1'], alpha=0.5)
plt.title('Scatter Plot of Pixel Values')
plt.xlabel('Pixel_0')
plt.ylabel('Pixel_1')
plt.show()

# Pair Plot (for demonstration purposes, considering only a subset of features)
subset_columns = ['Pixel_0', 'Pixel_10', 'Pixel_20', 'Pixel_30', 'Pixel_40']
sns.pairplot(data_df[subset_columns])
plt.show()
