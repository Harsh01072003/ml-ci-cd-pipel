import pandas as pd
from sklearn.datasets import fetch_openml

# Fetch the Boston House Price dataset from OpenML
boston = fetch_openml(name="boston", version=1, as_frame=True)

# Convert to DataFrame
df = boston.frame

# Save to CSV
df.to_csv(r'D:\ml-ci-cd-pipe\data\Boston.csv', index=False)

print("Boston dataset saved to CSV successfully.")
