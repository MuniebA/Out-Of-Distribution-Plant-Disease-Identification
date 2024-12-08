import zipfile
import pandas as pd
import os
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

class DatasetAnalyzer:
    def __init__(self, zip_path, train_csv_path, test_csv_path):
        self.zip_path = zip_path
        self.train_csv_path = train_csv_path
        self.test_csv_path = test_csv_path
        self.train_folder = 'Train'
        self.test_folder = 'Test'
        
        # Statistics storage
        self.zip_files = set()
        self.train_files = set()
        self.test_files = set()
        self.train_stats = defaultdict(int)
        self.test_stats = defaultdict(int)
    
    def normalize_path(self, path):
        """Convert path to use forward slashes."""
        return str(Path(path)).replace(os.sep, '/')
    
    def read_csv_file(self, file_path):
        """Read a comma-separated CSV file without headers."""
        try:
            df = pd.read_csv(file_path, header=None)
            column_names = ['file_path', 'Label', 'extra'] if len(df.columns) >= 3 else ['file_path', 'Label']
            df.columns = column_names
            return df
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")
            return None
    
    def analyze_dataset_distribution(self):
        """Analyze the distribution of images across datasets."""
        # Load zip contents
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            self.zip_files = {os.path.basename(name) for name in zip_ref.namelist() 
                            if name.lower().endswith(('.jpg', '.jpeg', '.png'))}
        
        # Load CSV files
        train_df = self.read_csv_file(self.train_csv_path)
        test_df = self.read_csv_file(self.test_csv_path)
        
        if train_df is not None:
            self.train_files = {os.path.basename(path) for path in train_df['file_path']}
            self.train_labels = train_df['Label'].value_counts().to_dict()
        
        if test_df is not None:
            self.test_files = {os.path.basename(path) for path in test_df['file_path']}
            self.test_labels = test_df['Label'].value_counts().to_dict()
        
        # Calculate distributions
        self.unaccounted_files = self.zip_files - (self.train_files | self.test_files)
        self.train_missing = self.train_files - self.zip_files
        self.test_missing = self.test_files - self.zip_files
    
    def create_visualizations(self):
        """Create comprehensive visualizations of the dataset distribution."""
        plt.rcParams['figure.figsize'] = [15, 10]
        plt.rcParams['figure.autolayout'] = True
        fig = plt.figure()
        
        # 1. Overall Dataset Distribution
        plt.subplot(2, 2, 1)
        distribution_data = {
            'Training CSV': len(self.train_files),
            'Test CSV': len(self.test_files),
            'Unaccounted': len(self.unaccounted_files)
        }
        
        bars = plt.bar(distribution_data.keys(), distribution_data.values(), 
                      color=['#2196F3', '#FFA726', '#FF5252'])
        plt.title('Complete Dataset Distribution', pad=20, fontsize=12, fontweight='bold')
        plt.ylabel('Number of Images')
        
        # Add percentage labels
        total_images = len(self.zip_files)
        for bar in bars:
            height = bar.get_height()
            percentage = (height / total_images) * 100
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}\n({percentage:.1f}%)',
                    ha='center', va='bottom')
        
        # 2. Training Labels Distribution
        plt.subplot(2, 2, 2)
        if self.train_labels:
            labels = list(self.train_labels.keys())
            values = list(self.train_labels.values())
            bars = plt.bar(labels, values, color='#2196F3')
            plt.title('Training Set Label Distribution', pad=20, fontsize=12, fontweight='bold')
            plt.xlabel('Label')
            plt.ylabel('Number of Images')
            plt.xticks(rotation=45)
            
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom')
        
        # 3. Test Labels Distribution
        plt.subplot(2, 2, 3)
        if self.test_labels:
            labels = list(self.test_labels.keys())
            values = list(self.test_labels.values())
            bars = plt.bar(labels, values, color='#FFA726')
            plt.title('Test Set Label Distribution', pad=20, fontsize=12, fontweight='bold')
            plt.xlabel('Label')
            plt.ylabel('Number of Images')
            plt.xticks(rotation=45)
            
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom')
        
        # 4. Dataset File Status
        plt.subplot(2, 2, 4)
        status_data = {
            'Training Set': [len(self.train_files - self.train_missing), len(self.train_missing)],
            'Test Set': [len(self.test_files - self.test_missing), len(self.test_missing)]
        }
        
        labels = list(status_data.keys())
        valid_files = [data[0] for data in status_data.values()]
        missing_files = [data[1] for data in status_data.values()]
        
        x = range(len(labels))
        plt.bar(x, valid_files, label='Found in Zip', color='#4CAF50')
        plt.bar(x, missing_files, bottom=valid_files, label='Missing from Zip', color='#FF5252')
        plt.title('Dataset File Status', pad=20, fontsize=12, fontweight='bold')
        plt.xlabel('Dataset')
        plt.ylabel('Number of Images')
        plt.xticks(x, labels)
        plt.legend()
        
        plt.tight_layout()
        
        # Save and show the plot
        try:
            fig.savefig('dataset_distribution.png', bbox_inches='tight', dpi=300)
            print("\nVisualizations saved as 'dataset_distribution.png'")
        except Exception as e:
            print(f"Error saving visualization: {str(e)}")
        
        plt.show()
        
        # Print detailed statistics
        print("\nDataset Statistics:")
        print(f"Total images in zip: {len(self.zip_files)}")
        print(f"Images in training CSV: {len(self.train_files)}")
        print(f"Images in test CSV: {len(self.test_files)}")
        print(f"Unaccounted images: {len(self.unaccounted_files)}")
        print(f"\nMissing Files:")
        print(f"Training files not in zip: {len(self.train_missing)}")
        print(f"Test files not in zip: {len(self.test_missing)}")
    
    def process_datasets(self, n_images=208):
        """Process and extract the datasets."""
        os.makedirs(self.train_folder, exist_ok=True)
        os.makedirs(self.test_folder, exist_ok=True)
        
        train_df = self.read_csv_file(self.train_csv_path)
        test_df = self.read_csv_file(self.test_csv_path)
        
        # Process zip file
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            # Process each dataset
            for df, output_folder, stats_dict in [
                (train_df, self.train_folder, self.train_stats),
                (test_df, self.test_folder, self.test_stats)
            ]:
                if df is None:
                    continue
                    
                label_counts = defaultdict(int)
                selected_files = []
                
                for _, row in df.iterrows():
                    file_path = row['file_path']
                    label = str(row['Label'])
                    base_name = os.path.basename(file_path)
                    
                    # Check if file exists in zip
                    if base_name in self.zip_files and label_counts[label] < n_images:
                        try:
                            # Find the full path in zip
                            zip_path = next(name for name in zip_ref.namelist() 
                                          if os.path.basename(name) == base_name)
                            
                            # Extract and process the file
                            zip_ref.extract(zip_path, output_folder)
                            final_path = os.path.join(output_folder, base_name)
                            
                            if os.path.exists(final_path):
                                label_counts[label] += 1
                                stats_dict[label] += 1
                                
                                selected_files.append({
                                    'file_path': self.normalize_path(f"{output_folder}/{base_name}"),
                                    'Label': label
                                })
                        except Exception as e:
                            print(f"Error processing {file_path}: {str(e)}")
                
                # Save processed files list
                if selected_files:
                    output_csv = f'{output_folder}_labels.csv'
                    pd.DataFrame(selected_files).to_csv(output_csv, index=False)
                    print(f"\nCreated {output_csv} with {len(selected_files)} entries")
    
    def run_analysis(self):
        """Run the complete dataset analysis and processing pipeline."""
        print("Analyzing dataset distribution...")
        self.analyze_dataset_distribution()
        
        print("\nProcessing datasets...")
        self.process_datasets()
        
        print("\nGenerating visualizations...")
        self.create_visualizations()

# Example usage
if __name__ == "__main__":
    analyzer = DatasetAnalyzer(
        zip_path="plantvillage.zip",
        train_csv_path="PV train.csv",
        test_csv_path="PV test seen.csv"
    )
    analyzer.run_analysis()