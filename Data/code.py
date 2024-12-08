import os
import shutil
import pandas as pd
from pathlib import Path


def create_subset_dataset():
    # Define paths
    original_csv = 'Train_labels.csv'
    original_image_dir = 'Train'
    output_dir_crop = 'subset_dataset_crop'
    output_dir_disease = 'subset_dataset_disease'
    output_csv = 'subset_labels.csv'

    # Create output directories if they don't exist
    os.makedirs(output_dir_crop, exist_ok=True)
    os.makedirs(output_dir_disease, exist_ok=True)

    # Read the CSV file without headers
    df = pd.read_csv(original_csv, header=None, names=[
                     'filename', 'crop_label', 'disease_label'])

    # Initialize lists to store selected data
    selected_rows = []
    processed_crops = set()
    processed_diseases = set()

    # First pass: Process crops
    for _, row in df.iterrows():
        crop_label = row['crop_label']

        # Check if we need this crop
        if crop_label not in processed_crops and crop_label < 14:
            # Add to processed crops
            processed_crops.add(crop_label)

            # Copy and rename the image for crop folder
            original_image_path = os.path.join(
                original_image_dir, row['filename'])
            if os.path.exists(original_image_path):
                new_filename = f"crop_{crop_label}.jpg"
                new_image_path = os.path.join(output_dir_crop, new_filename)
                shutil.copy2(original_image_path, new_image_path)

                # Store row information
                crop_row = row.copy()
                crop_row['filename'] = new_filename
                selected_rows.append(crop_row)

        # Check if we have all required crops
        if len(processed_crops) >= 14:
            break

    # Second pass: Process diseases
    for _, row in df.iterrows():
        disease_label = row['disease_label']

        # Check if we need this disease
        if disease_label not in processed_diseases and disease_label < 21:
            # Add to processed diseases
            processed_diseases.add(disease_label)

            # Copy and rename the image for disease folder
            original_image_path = os.path.join(
                original_image_dir, row['filename'])
            if os.path.exists(original_image_path):
                new_filename = f"disease_{disease_label}.jpg"
                new_image_path = os.path.join(output_dir_disease, new_filename)
                shutil.copy2(original_image_path, new_image_path)

                # Store row information
                disease_row = row.copy()
                disease_row['filename'] = new_filename
                selected_rows.append(disease_row)

        # Check if we have all required diseases
        if len(processed_diseases) >= 21:
            break

    # Create new DataFrame with selected rows
    subset_df = pd.DataFrame(selected_rows)

    # Save the new CSV file
    subset_df.to_csv(output_csv, index=False)

    # Print summary
    print(f"\nCreated subset datasets:")
    print(f"Crop folder ({output_dir_crop}):")
    print(f"- Number of unique crops: {len(processed_crops)}")
    print(f"- Processed crop labels: {sorted(list(processed_crops))}")

    print(f"\nDisease folder ({output_dir_disease}):")
    print(f"- Number of unique diseases: {len(processed_diseases)}")
    print(f"- Processed disease labels: {sorted(list(processed_diseases))}")

    print(f"\nTotal images processed: {len(selected_rows)}")
    print(f"Output CSV file: {output_csv}")


if __name__ == "__main__":
    create_subset_dataset()
