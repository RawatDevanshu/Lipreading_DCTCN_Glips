import os
import pandas as pd

# Define the directory path where the GLIPS dataset resides
glips_dataset_path = "path-to-dataset"

# Create an empty list to store file paths
file_paths = []

# Recursively walk through the directory and collect file paths
for filename in os.listdir(glips_dataset_path):
    for split in ["train","test","val"]:
        for video_filename in os.listdir(os.path.join(glips_dataset_path,filename,split)):
            if not video_filename.endswith('.mp4'):
                continue
            file_paths.append(os.path.join(filename,split,video_filename[:-4]))
        

# Create a Pandas DataFrame with one column named "path"
df = pd.DataFrame({"path": file_paths})

# Specify the output filename for the CSV file
output_filename = "glips_dataset_paths.csv"

# Save the DataFrame to a CSV file
df.to_csv(output_filename, index=False)

print(f"CSV file generated successfully! File saved as: {output_filename}")
