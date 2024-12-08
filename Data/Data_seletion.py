import zipfile
import pandas as pd
import os
from collections import defaultdict
from pathlib import Path
import numpy as np


def normalize_path(path):
    """Convert path to use forward slashes."""
    return str(Path(path)).replace(os.sep, '/')


def read_custom_csv(csv_path):
    """
    Read CSV file with format:
    image_name,crop_class,disease_class
    Each line contains: filename,crop_num,disease_num
    """
    try:
        data = []
        with open(csv_path, 'r') as file:
            for line in file:
                # Split each line by comma
                parts = line.strip().split(',')
                if len(parts) == 3:  # Verify we have all three components
                    try:
                        image_name = parts[0].strip()
                        crop = int(parts[1].strip())
                        disease = int(parts[2].strip())
                        data.append({
                            'image_name': image_name,
                            'crop': crop,
                            'disease': disease
                        })
                    except ValueError as e:
                        print(f"Warning: Invalid numeric values in line: {
                              line.strip()}")
                        continue
                else:
                    print(f"Warning: Malformed line (expected 3 parts, got {
                          len(parts)}): {line.strip()}")

        # Create DataFrame
        df = pd.DataFrame(data)

        print(f"\nSuccessfully read {csv_path}")
        print(f"Found {len(df)} rows")
        if not df.empty:
            print("\nSample of first few rows:")
            print(df.head())
            print("\nCrop distribution:")
            print(df['crop'].value_counts().sort_index())
            print("\nDisease distribution:")
            print(df['disease'].value_counts().sort_index())

        return df

    except Exception as e:
        print(f"Error reading {csv_path}: {str(e)}")
        # Print the first few lines of the file for debugging
        try:
            with open(csv_path, 'r') as file:
                print(f"\nFirst few lines of {csv_path}:")
                for i, line in enumerate(file):
                    if i < 5:  # Print first 5 lines
                        print(f"Line {i+1}: {line.strip()}")
                    else:
                        break
        except Exception as read_error:
            print(f"Could not read file content: {str(read_error)}")
        return pd.DataFrame(columns=['image_name', 'crop', 'disease'])
    

def print_dataset_info(df, dataset_name):
    """Print detailed information about the dataset."""
    print(f"\nDataset: {dataset_name}")
    print(f"Total samples: {len(df)}")
    if not df.empty:
        print("\nCrop distribution:")
        print(df['crop'].value_counts().sort_index())
        print("\nDisease distribution:")
        print(df['disease'].value_counts().sort_index())
        print("\nSample of data:")
        print(df.head())
        print("\nComposition counts:")
        print(df.groupby(['crop', 'disease']).size(
        ).sort_values(ascending=False).head(10))


def process_multiple_datasets(config):
    """Process multiple datasets with balanced sampling."""
    # Create output directories
    output_dirs = {
        'train': 'Train',
        'test_seen': 'Test_Seen',
        'test_unseen': 'Test_Unseen',
        'pd_test': 'PD_Test'
    }

    for dir_name in output_dirs.values():
        os.makedirs(dir_name, exist_ok=True)
        print(f"Created directory: {dir_name}/")

    # Read all CSV files
    print("\nReading CSV files...")
    dfs = {
        'train': read_custom_csv(config['train_csv']),
        'test_seen': read_custom_csv(config['test_seen_csv']),
        'test_unseen': read_custom_csv(config['test_unseen_csv']),
        'pd_test': read_custom_csv(config['pd_test_csv'])
    }

    # Print initial dataset statistics
    for name, df in dfs.items():
        print_dataset_info(df, name)

    # Set sample sizes
    sample_sizes = {
        'train': 200,
        'test_seen': 50,
        'test_unseen': 50,
        'pd_test': 50
    }

    # Process each dataset
    try:
        zip_files = {
            'pv': zipfile.ZipFile(config['pv_zip_path'], 'r'),
            'pd': zipfile.ZipFile(config['pd_zip_path'], 'r')
        }
        print("\nOpened zip files successfully")

        # Print contents of zip files for debugging
        print("\nPlantVillage zip contents (first 5 files):")
        pv_files = zip_files['pv'].namelist()
        print("\n".join(pv_files[:5]))

        print("\nPlantDoc zip contents (first 5 files):")
        pd_files = zip_files['pd'].namelist()
        print("\n".join(pd_files[:5]))

    except FileNotFoundError as e:
        print(f"Error opening zip files: {str(e)}")
        return

    try:
        for dataset_name, df in dfs.items():
            if df.empty:
                print(f"\nSkipping {dataset_name} dataset (empty)")
                continue

            print(f"\nProcessing {dataset_name} dataset...")
            output_dir = output_dirs[dataset_name]

            # Determine which zip file to use
            zip_file = zip_files['pd'] if dataset_name == 'pd_test' else zip_files['pv']
            zip_files_set = set(zip_file.namelist())

            # Group by crop-disease combination and sample
            processed_files = []

            for (crop, disease), group in df.groupby(['crop', 'disease']):
                print(f"\nProcessing crop {crop}, disease {disease}")
                print(f"Found {len(group)} images in this combination")

                # Sample images
                n_samples = min(len(group), sample_sizes[dataset_name])
                sampled_rows = group.sample(n=n_samples, random_state=42)

                for _, row in sampled_rows.iterrows():
                    image_name = row['image_name']
                    print(f"Processing image: {image_name}")

                    # Try different possible paths
                    possible_paths = [
                        image_name,
                        f"plantvillage/{image_name}",
                        image_name.replace('plantvillage/', ''),
                        f"PlantVillage/{image_name}",
                        image_name.replace('PlantVillage/', '')
                    ]

                    # Try both .jpg and .JPG variations for each path
                    all_paths = []
                    for path in possible_paths:
                        all_paths.extend([
                            path,
                            path.replace('.JPG', '.jpg'),
                            path.replace('.jpg', '.JPG')
                        ])

                    # Find the first matching path
                    zip_path_found = next(
                        (p for p in all_paths if p in zip_files_set), None)

                    if zip_path_found:
                        try:
                            print(f"Found image at path: {zip_path_found}")
                            # Extract the file
                            zip_file.extract(zip_path_found, output_dir)

                            # Get the final filename
                            final_filename = os.path.basename(zip_path_found)

                            # Handle potential nested directories
                            extracted_path = os.path.join(
                                output_dir, zip_path_found)
                            final_path = os.path.join(
                                output_dir, final_filename)

                            # Move file if it was extracted to a subdirectory
                            if os.path.dirname(extracted_path) != output_dir:
                                if not os.path.exists(output_dir):
                                    os.makedirs(output_dir)
                                if os.path.exists(extracted_path):
                                    os.rename(extracted_path, final_path)

                            processed_files.append(
                                [final_filename, crop, disease])
                            print(f"Successfully processed {final_filename}")
                        except Exception as e:
                            print(f"Error processing {
                                  zip_path_found}: {str(e)}")
                    else:
                        print(f"Warning: Could not find image {
                              image_name} in zip file")

            # Create the output CSV file
            if processed_files:
                output_csv = f'{output_dir}_labels.csv'
                print(f"\nWriting {len(processed_files)
                                   } entries to {output_csv}")

                with open(output_csv, 'w') as f:
                    for file_info in processed_files:
                        f.write(f"{file_info[0]} {
                                file_info[1]} {file_info[2]}\n")

                print(f"Successfully created {output_csv}")

                # Print final statistics
                print(f"\nFinal statistics for {dataset_name}:")
                print(f"Total images processed: {len(processed_files)}")
                unique_crops = len(set(row[1] for row in processed_files))
                unique_diseases = len(set(row[2] for row in processed_files))
                print(f"Unique crops: {unique_crops}")
                print(f"Unique diseases: {unique_diseases}")
            else:
                print(f"\nNo files were processed for {dataset_name}")

    finally:
        # Clean up
        for zip_file in zip_files.values():
            zip_file.close()

        # Clean up any empty directories
        for output_dir in output_dirs.values():
            for root, dirs, _ in os.walk(output_dir, topdown=False):
                for name in dirs:
                    try:
                        dir_path = os.path.join(root, name)
                        if not os.listdir(dir_path):
                            os.rmdir(dir_path)
                    except OSError:
                        pass


if __name__ == "__main__":
    config = {
        'pv_zip_path': "plantvillage.zip",
        'pd_zip_path': "plantdoc.zip",
        'train_csv': "PV train.csv",
        'test_seen_csv': "PV test seen.csv",
        'test_unseen_csv': "PV test unseen.csv",
        'pd_test_csv': "PD test unseen.csv"
    }

    process_multiple_datasets(config)
