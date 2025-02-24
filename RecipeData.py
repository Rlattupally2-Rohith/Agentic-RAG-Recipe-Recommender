import pandas as pd
import kagglehub
import zipfile
import os
import shutil


path = kagglehub.dataset_download("shuyangli94/food-com-recipes-and-user-interactions")
print("Path to downloaded ZIP file:", path)


new_path = os.path.join(os.path.expanduser("~"), "Desktop", "food_com_dataset.zip")
shutil.copy(path, new_path)
print("File copied to:", new_path)


extract_dir = os.path.join(os.path.expanduser("~"), "Desktop", "food_com_dataset")
if not os.path.exists(extract_dir):
    os.makedirs(extract_dir)


try:
    with zipfile.ZipFile(new_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Dataset extracted to:", extract_dir)
except zipfile.BadZipFile:
    print("Error: The downloaded file is not a valid ZIP file.")
    exit()


dataset_files = os.listdir(extract_dir)
print("Dataset files:", dataset_files)

try:
    recipes_df = pd.read_csv(os.path.join(extract_dir, "RAW_recipes.csv"))
    print("Dataset loaded successfully!")
    print(recipes_df.head())
except FileNotFoundError:
    print("Error: RAW_recipes.csv not found in the extracted files.")


