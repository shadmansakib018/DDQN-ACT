import os
import shutil
import pandas as pd
from datetime import datetime


def makeResults():
    # Base destination folder
    base_results_dir = r"C:\Users\ss4587s\Desktop\CloudSimCSVs\RL"

    # Create a dynamic subfolder name based on current date and time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp2 = datetime.now().strftime("%H_%M")
    destination_dir = os.path.join(base_results_dir, f"results_{timestamp}")
    os.makedirs(destination_dir, exist_ok=True)

    # Source directory
    source_dir = r"C:\Users\ss4587s\Desktop\CloudSimCSVs\RL\EXP1"
    csv_output_path = os.path.join(destination_dir, "all_results_" + timestamp2 + ".csv")

    # Dictionary to collect all data
    data_dict = {}

    # Move .txt files and parse data
    for filename in os.listdir(source_dir):
        if filename.endswith(".txt"):
            # Full paths
            src_path = os.path.join(source_dir, filename)
            dst_path = os.path.join(destination_dir, filename)
            
            # Move the file
            shutil.move(src_path, dst_path)
            
            # Extract experiment name (remove .txt)
            experiment_name = os.path.splitext(filename)[0]

            # Read all lines and parse as floats
            with open(dst_path, "r") as f:
                lines = [float(line.strip()) for line in f if line.strip()]
            
            # Store in dictionary
            data_dict[experiment_name] = lines

    # Create DataFrame and export to CSV
    df = pd.DataFrame(data_dict)  # Sorted for consistency
    df.to_csv(csv_output_path, index=False)

    print(f"✅ Results saved to {csv_output_path}")

# makeResults()