import os

print("Running Data Pipeline...")

os.system("python3 data_ingestion.py")
os.system("python3 data_preprocessing.py")
os.system("python3 data_transformation.py")
os.system("python3 data_merge.py")
os.system("python3 data_saving.py")

print("✅ Data Pipeline Completed Successfully!")
