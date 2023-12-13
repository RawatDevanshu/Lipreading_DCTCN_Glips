import os
import pandas as pd

# Define the directory path where the GLIPS dataset resides
glips_dataset_path = "Path-to-dataset"

# Create an empty list to store file paths
words = []

count = 0
# Recursively walk through the directory and collect file paths
for filename in os.listdir(glips_dataset_path):
    words.append(os.path.join(filename))
    count += 1
        

# Create a Pandas DataFrame with one column named "path"
df = pd.DataFrame({"words": words})

# Specify the output filename for the CSV file
output_filename = f"{count}WordsSortedList_Glips.csv"

# Save the DataFrame to a CSV file
df.to_csv(output_filename, index=False)

print(f"CSV file generated successfully! File saved as: {output_filename}")
