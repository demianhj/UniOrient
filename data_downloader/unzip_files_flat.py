import os
import tarfile

def extract_tar_gz_to_named_folder(tar_path):
    # Determine extraction directory from tar name
    base_name = os.path.splitext(os.path.splitext(os.path.basename(tar_path))[0])[0]
    extract_dir = os.path.join(os.path.dirname(tar_path), base_name)

    # If already extracted (folder exists and not empty), skip extraction
    if os.path.isdir(extract_dir) and os.listdir(extract_dir):
        print(f"Skipping extraction (already done): {tar_path}")
        # But still ensure the archive is replaced by an empty file
        if os.path.exists(tar_path) and os.path.getsize(tar_path) > 0:
            print(f"Removing leftover archive and creating empty placeholder: {tar_path}")
            os.remove(tar_path)
            # open(tar_path, 'a').close()
        return

    print(f"Extracting: {tar_path} -> {extract_dir}")
    os.makedirs(extract_dir, exist_ok=True)

    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)
        # Remove the original archive
        os.remove(tar_path)
        # Recreate an empty file with the same name
        # open(tar_path, 'a').close()
    except Exception as e:
        print(f"Failed to extract {tar_path}: {e}")

def process_category(category_dir):
    if  os.path.isdir(category_dir):
        
        
        for subdir in os.listdir(category_dir):
            print("show category_dir: ", subdir)
            subdir_path = os.path.join(category_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue

            for file in os.listdir(subdir_path):
                if file.endswith(".tar.gz"):
                    tar_path = os.path.join(subdir_path, file)
                    extract_tar_gz_to_named_folder(tar_path)

                elif file.endswith(".tar.gz.1"):  # ".1"-Versionen ignorieren/löschen
                    tar_path = os.path.join(subdir_path, file)
                    os.remove(tar_path)
                    print(f"Removed duplicate archive: {tar_path}")

if __name__ == "__main__":
    base_dir = "/home/xyz/student/lei/shape_canonicalization_gui_tool/data/G-objaverse"

    # print("show base_dir:",base_dir)
    # exit(0)
    for folder in  os.listdir(base_dir):
        category_dir = os.path.join(base_dir, folder)
        process_category(category_dir)
