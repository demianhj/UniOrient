import os

# Change this to the path you want to scan
base_path = "/home/xyz/student/lei/shape_canonicalization_gui_tool/data/G-objaverse_h5py_files_v1_results"

# File to save folder names
output_file = "2folders.txt"

# Get all folder names
folders = [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]

# Save to text file
with open(output_file, "w") as f:
    for folder in folders:
        f.write(folder + "\n")

print(f"Saved {len(folders)} folder names to {output_file}")

