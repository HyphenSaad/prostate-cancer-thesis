import pandas as pd
import os

selected_files = []
for root, dirs, files in os.walk('data/slides'):
    for file in files:
        if file.endswith('.tiff'):
            selected_files.append(file.split('.')[0])

df = pd.read_csv("data/train.csv")
df = df[df['image_id'].isin(selected_files)]
df.to_csv('data/dataset_info.csv', index=False)