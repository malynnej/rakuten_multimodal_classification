import os

paths = [
    "./data/raw/images/image_train/"
]


def create_folder_structure(path):

    for path in paths:
        if not os.path.exists(path) or not os.path.isdir(path):
                os.makedirs(path)
                print(f"Path: {path} created successfully.")
        else:
             print(f"Path: {path} already exists.")

file_paths = [
     "./data/raw/PLACEHOLDER_X_train.txt",
     "./data/raw/PLACEHOLDER_y_train.txt",
     "./data/raw/images/image_train/PLACEHOLDER_images.txt"
]

def create_placeholer(file_paths):

    for file_path in file_paths:
        with open(file_path, "w", encoding='utf-8') as file:
            file.write(file_path.split("/")[-1])
        
        print(f"File: {file_path} created successfully.")

create_folder_structure(paths)
create_placeholer(file_paths)