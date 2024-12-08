import zipfile
import pandas as pd
import os
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt


class DatasetAnalyzer:
    def __init__(self, zip_paths, csv_paths):
        self.zip_paths = zip_paths  # List of zip file paths
        self.csv_paths = csv_paths  # List of CSV file paths

        # Statistics storage
        self.zip_files = set()
        self.zip_file_images = {}  # Map zip file names to sets of images
        self.image_to_zip = {}  # Map image filenames to the zip file they are in
        self.csv_files = defaultdict(set)
        self.csv_labels = {}
        # Map image filenames to CSV files
        self.image_to_csv = defaultdict(set)
        # Map zip files to unaccounted images
        self.unaccounted_files = defaultdict(set)
        self.missing_files = defaultdict(set)
        self.stats = defaultdict(lambda: defaultdict(int))

        # Folders for datasets
        self.folders = {}
        for csv_path in self.csv_paths:
            dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
            self.folders[dataset_name] = dataset_name
            os.makedirs(dataset_name, exist_ok=True)

    def normalize_path(self, path):
        """Convert path to use forward slashes."""
        return str(Path(path)).replace(os.sep, '/')

    def read_csv_file(self, file_path):
        """Read a comma-separated CSV file without headers."""
        try:
            df = pd.read_csv(file_path, header=None)
            column_names = ['file_path', 'Label', 'extra'] if len(df.columns) >= 3 else [
                'file_path', 'Label']
            df.columns = column_names
            return df
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")
            return None

    def analyze_dataset_distribution(self):
        """Analyze the distribution of images across datasets."""
        # Load zip contents from all zip files
        for zip_path in self.zip_paths:
            zip_name = os.path.basename(zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                image_files = {os.path.basename(name) for name in zip_ref.namelist()
                               if name.lower().endswith(('.jpg', '.jpeg', '.png'))}
                self.zip_files.update(image_files)
                self.zip_file_images[zip_name] = image_files
                for image in image_files:
                    self.image_to_zip[image] = zip_name

        # Load CSV files
        for csv_path in self.csv_paths:
            df = self.read_csv_file(csv_path)
            dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
            if df is not None:
                image_files = {os.path.basename(path)
                               for path in df['file_path']}
                self.csv_files[dataset_name] = image_files
                self.csv_labels[dataset_name] = df['Label'].value_counts(
                ).to_dict()
                for image in image_files:
                    self.image_to_csv[image].add(dataset_name)
            else:
                self.csv_files[dataset_name] = set()
                self.csv_labels[dataset_name] = {}

        # Calculate distributions
        all_csv_files = set().union(*self.csv_files.values())
        unaccounted_images = self.zip_files - all_csv_files
        # Map unaccounted images to zip files
        for image in unaccounted_images:
            zip_name = self.image_to_zip.get(image)
            if zip_name:
                self.unaccounted_files[zip_name].add(image)

        # Missing files per dataset
        for dataset_name, files in self.csv_files.items():
            missing = files - self.zip_files
            self.missing_files[dataset_name] = missing

    def create_visualizations(self):
        """Create comprehensive visualizations of the dataset distribution."""
        plt.rcParams['figure.figsize'] = [20, 15]
        plt.rcParams['figure.autolayout'] = True
        fig = plt.figure()

        # 1. Overall Dataset Distribution
        plt.subplot(2, 2, 1)
        distribution_data = {name: len(files)
                             for name, files in self.csv_files.items()}
        total_unaccounted = sum(len(files)
                                for files in self.unaccounted_files.values())
        distribution_data.update({f"Unaccounted ({zip_name})": len(files)
                                  for zip_name, files in self.unaccounted_files.items()})

        colors = ['#2196F3', '#FFA726', '#FF5252',
                  '#AB47BC', '#26A69A', '#8D6E63', '#FFAB91']
        bars = plt.bar(distribution_data.keys(), distribution_data.values(),
                       color=colors[:len(distribution_data)])
        plt.title('Complete Dataset Distribution',
                  pad=20, fontsize=14, fontweight='bold')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=45)

        # Add percentage labels
        total_images = len(self.zip_files)
        for bar in bars:
            height = bar.get_height()
            percentage = (height / total_images) * \
                100 if total_images > 0 else 0
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{int(height)}\n({percentage:.1f}%)',
                     ha='center', va='bottom', fontsize=10)

        # 2. Labels Distribution for PV Train
        plt.subplot(2, 2, 2)
        dataset_name = 'PV train'
        if dataset_name in self.csv_labels:
            label_counts = self.csv_labels[dataset_name]
            labels = list(label_counts.keys())
            values = list(label_counts.values())
            bars = plt.bar(labels, values, color='#2196F3')
            plt.title('PV Train Label Distribution', pad=20,
                      fontsize=14, fontweight='bold')
            plt.xlabel('Label')
            plt.ylabel('Number of Images')
            plt.xticks(rotation=45)

            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                         f'{int(height)}',
                         ha='center', va='bottom', fontsize=10)

        # 3. Labels Distribution for PV Test Seen
        plt.subplot(2, 2, 3)
        dataset_name = 'PV test seen'
        if dataset_name in self.csv_labels:
            label_counts = self.csv_labels[dataset_name]
            labels = list(label_counts.keys())
            values = list(label_counts.values())
            bars = plt.bar(labels, values, color='#FFA726')
            plt.title('PV Test Seen Label Distribution',
                      pad=20, fontsize=14, fontweight='bold')
            plt.xlabel('Label')
            plt.ylabel('Number of Images')
            plt.xticks(rotation=45)

            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                         f'{int(height)}',
                         ha='center', va='bottom', fontsize=10)

        # 4. Labels Distribution for PV Test Unseen and PD Test Unseen
        plt.subplot(2, 2, 4)
        datasets_to_plot = ['PV test unseen', 'PD test unseen']
        total_labels = set()
        for dataset_name in datasets_to_plot:
            if dataset_name in self.csv_labels:
                total_labels.update(self.csv_labels[dataset_name].keys())
        total_labels = sorted(total_labels)
        bar_width = 0.35
        x = range(len(total_labels))
        for idx, dataset_name in enumerate(datasets_to_plot):
            if dataset_name in self.csv_labels:
                label_counts = self.csv_labels[dataset_name]
                values = [label_counts.get(label, 0) for label in total_labels]
                positions = [i + idx * bar_width for i in x]
                bars = plt.bar(positions, values, width=bar_width,
                               label=dataset_name, color=colors[idx+2])
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        plt.text(bar.get_x() + bar.get_width()/2., height,
                                 f'{int(height)}',
                                 ha='center', va='bottom', fontsize=10)
        plt.xlabel('Label')
        plt.ylabel('Number of Images')
        plt.xticks([i + bar_width / 2 for i in x], total_labels, rotation=45)
        plt.title('Unseen Test Sets Label Distribution',
                  pad=20, fontsize=14, fontweight='bold')
        plt.legend()

        plt.tight_layout()

        # Save and show the plot
        try:
            fig.savefig('dataset_distribution.png',
                        bbox_inches='tight', dpi=300)
            print("\nVisualizations saved as 'dataset_distribution.png'")
        except Exception as e:
            print(f"Error saving visualization: {str(e)}")

        plt.show()

        # Print detailed statistics
        print("\nDataset Statistics:")
        print(f"Total images in zip files: {len(self.zip_files)}")
        for dataset_name, files in self.csv_files.items():
            print(f"Images in {dataset_name}: {len(files)}")
        for zip_name, files in self.unaccounted_files.items():
            print(f"Unaccounted images in {zip_name}: {len(files)}")
        print(f"\nMissing Files:")
        for dataset_name, missing in self.missing_files.items():
            print(f"Files in {dataset_name} not in zip files: {len(missing)}")

    def process_datasets(self, n_images_per_label=208):
        """Process and extract the datasets."""
        # Read CSV files
        dfs = {}
        for csv_path in self.csv_paths:
            dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
            df = self.read_csv_file(csv_path)
            dfs[dataset_name] = df

        # Create a mapping of filename to zip file path
        file_zip_map = {}
        for zip_path in self.zip_paths:
            zip_name = os.path.basename(zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for name in zip_ref.namelist():
                    if name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        base_name = os.path.basename(name)
                        # Allow multiple files with the same name from different zips
                        if base_name not in file_zip_map:
                            file_zip_map[base_name] = (zip_ref.filename, name)

        # Process each dataset
        for dataset_name, df in dfs.items():
            if df is None:
                continue
            output_folder = self.folders[dataset_name]
            stats_dict = self.stats[dataset_name]

            label_counts = defaultdict(int)
            selected_files = []

            for _, row in df.iterrows():
                file_path = row['file_path']
                label = str(row['Label'])
                base_name = os.path.basename(file_path)

                # Check if file exists in the zip files
                if base_name in file_zip_map and label_counts[label] < n_images_per_label:
                    zip_filename, zip_file_path = file_zip_map[base_name]
                    try:
                        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                            # Extract and process the file
                            zip_ref.extract(zip_file_path, output_folder)
                            final_path = os.path.join(
                                output_folder, os.path.basename(zip_file_path))

                            if os.path.exists(final_path):
                                label_counts[label] += 1
                                stats_dict[label] += 1

                                selected_files.append({
                                    'file_path': self.normalize_path(f"{output_folder}/{os.path.basename(zip_file_path)}"),
                                    'Label': label,
                                    'ZipFile': os.path.basename(zip_filename)
                                })
                    except Exception as e:
                        print(f"Error processing {file_path}: {str(e)}")
                else:
                    # File not found in any zip
                    pass

            # Save processed files list
            if selected_files:
                output_csv = f'{output_folder}_labels.csv'
                pd.DataFrame(selected_files).to_csv(output_csv, index=False)
                print(f"\nCreated {output_csv} with {
                      len(selected_files)} entries")

    def run_analysis(self):
        """Run the complete dataset analysis and processing pipeline."""
        print("Analyzing dataset distribution...")
        self.analyze_dataset_distribution()

        # Uncomment the following lines if you want to process datasets
        # print("\nProcessing datasets...")
        # self.process_datasets()

        print("\nGenerating visualizations...")
        self.create_visualizations()


# Example usage
if __name__ == "__main__":
    analyzer = DatasetAnalyzer(
        zip_paths=["plantvillage.zip", "plantdoc.zip"],
        csv_paths=["PV train.csv", "PV test seen.csv",
                   "PV test unseen.csv", "PD test unseen.csv"]
    )
    analyzer.run_analysis()
