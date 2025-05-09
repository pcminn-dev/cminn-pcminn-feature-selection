
#!/usr/bin/env python
# Download Bankruptcy Dataset from Kaggle

import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# Set up Kaggle API
api = KaggleApi()
api.authenticate()

# Create a data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Download the dataset
print("⬇️ Downloading Bankruptcy dataset from Kaggle...")
api.dataset_download_files('fedesoriano/company-bankruptcy-prediction', path='data', unzip=True)

# Check if file exists and rename if needed
csv_path = os.path.join("data", "data.csv")
if not os.path.exists(csv_path):
    # Try to find the actual CSV
    for fname in os.listdir("data"):
        if fname.endswith(".csv"):
            os.rename(os.path.join("data", fname), csv_path)
            break

print("✅ Download complete. Dataset is ready as data/data.csv")
