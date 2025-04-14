import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Read the data
df = pd.read_csv('nyc-rolling-sales.csv')

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Create cross-product feature between borough and neighborhood
df['borough_neighborhood'] = df['borough'].astype(str) + '_' + df['neighborhood'].astype(str)
le = LabelEncoder()
df['borough_neighborhood'] = le.fit_transform(df['borough_neighborhood'])

# Save the updated dataset
df.to_csv('nyc-rolling-sales-with-cross-product.csv', index=False)

print("Cross-product feature created and saved successfully!")
print(f"Number of unique borough-neighborhood combinations: {df['borough_neighborhood'].nunique()}") 