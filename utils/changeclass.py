import os
import re

# Directory path (change this to your target folder path)
folder_path = '../data/train'  # Current directory, modify as needed

# List of classes to keep
classes_to_keep = ['IDF', 'F15', 'F18']

# Regular expression to extract class name from filename
# Assuming format like "EA-18G_hrrp_theta_90_phi_0.4.mat"
pattern = r'^([^_]+)_'

# Counter for statistics
kept_files = 0
deleted_files = 0

# Scan all .mat files in the directory
for filename in os.listdir(folder_path):
    if filename.endswith('.mat'):
        # Extract class name using regex
        match = re.match(pattern, filename)
        if match:
            class_name = match.group(1)

            # Check if class should be kept
            if class_name in classes_to_keep:
                print(f"Keeping: {filename}")
                kept_files += 1
            else:
                # Delete file
                file_path = os.path.join(folder_path, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {filename}")
                    deleted_files += 1
                except Exception as e:
                    print(f"Error deleting {filename}: {e}")

# Print summary
print(f"\nSummary: Kept {kept_files} files, Deleted {deleted_files} files")
print(f"Remaining classes: {', '.join(classes_to_keep)}")