from src.train_pipeline import train_student_success_model
import os

if __name__ == "__main__":
    # Ensure we are in the root directory
    print(f"Current working directory: {os.getcwd()}")
    
    # Run the pipeline
    # Assuming dataset.csv is in the root
    data_path = 'dataset.csv'
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found!")
    else:
        model, results = train_student_success_model(data_path, save_model=True)
        print("Pipeline finished successfully.")
