# **LIBRARY IMPORTS**

import os
import zipfile
import pandas as pd
import numpy as np
from google.colab import drive
from pathlib import Path
from typing import Tuple, Dict, List, Optional

# Image processing
import cv2
from PIL import Image
from tqdm.notebook import tqdm

# Deep learning
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import timm
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import WeightedRandomSampler

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Utilities
from sklearn.model_selection import train_test_split
import logging
import warnings
import time
from collections import defaultdict
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# **DATA LOADING AND PREPARATION**
## Dataset Loading and Validation


import os
import zipfile
import logging
from pathlib import Path
from typing import Dict, Tuple
from google.colab import drive

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetLoader:
    """
    Handles dataset loading and validation for the Bird Classification project.

    Attributes:
        drive_path (str): Path to the Google Drive folder
        dataset_name (str): Name of the dataset zip file
        extract_path (str): Path where dataset will be extracted
    """

    def __init__(self):
        # Initialize paths
        self.drive_path = '/content/drive/MyDrive/Crop Identification'
        self.dataset_name = 'Dataset.zip'
        self.extract_base = '/content/dataset'

        # Full paths after extraction
        self.dataset_root = os.path.join(self.extract_base, 'Dataset')
        self.paths = {
            'zip_file': os.path.join(self.drive_path, self.dataset_name),
            'train_dir': os.path.join(self.dataset_root, 'Train'),
            'test_dir': os.path.join(self.dataset_root, 'Test'),
            'train_labels': os.path.join(self.dataset_root, 'train_labels.csv'),
            'test_labels': os.path.join(self.dataset_root, 'test_labels.csv')
        }

    def mount_drive(self) -> bool:
        """Mount Google Drive."""
        try:
            logger.info("Mounting Google Drive...")
            drive.mount('/content/drive', force_remount=True)
            return True
        except Exception as e:
            logger.error(f"Failed to mount drive: {str(e)}")
            return False

    def extract_dataset(self) -> bool:
        """Extract dataset from zip file."""
        try:
            if not os.path.exists(self.paths['zip_file']):
                raise FileNotFoundError(f"Dataset zip not found at {self.paths['zip_file']}")

            logger.info("Extracting dataset...")
            with zipfile.ZipFile(self.paths['zip_file'], 'r') as zip_ref:
                zip_ref.extractall(self.extract_base)
            logger.info("Dataset extracted successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to extract dataset: {str(e)}")
            return False

    def validate_dataset(self) -> Dict[str, Dict[str, any]]:
        """
        Validate the extracted dataset structure and contents.

        Returns:
            Dictionary containing validation results for each component
        """
        validation = {
            'directories': {
                'train': {'exists': False, 'count': 0},
                'test': {'exists': False, 'count': 0}
            },
            'labels': {
                'train': {'exists': False, 'size': 0},
                'test': {'exists': False, 'size': 0}
            }
        }

        # Validate directories and count images
        if os.path.exists(self.paths['train_dir']):
            validation['directories']['train']['exists'] = True
            validation['directories']['train']['count'] = len([
                f for f in os.listdir(self.paths['train_dir'])
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])

        if os.path.exists(self.paths['test_dir']):
            validation['directories']['test']['exists'] = True
            validation['directories']['test']['count'] = len([
                f for f in os.listdir(self.paths['test_dir'])
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])

        # Validate label files
        if os.path.exists(self.paths['train_labels']):
            validation['labels']['train']['exists'] = True
            validation['labels']['train']['size'] = os.path.getsize(self.paths['train_labels'])

        if os.path.exists(self.paths['test_labels']):
            validation['labels']['test']['exists'] = True
            validation['labels']['test']['size'] = os.path.getsize(self.paths['test_labels'])

        return validation

    def print_validation_results(self, validation: Dict) -> None:
        """Print formatted validation results."""
        print("\nDataset Validation Results:")
        print("-" * 50)

        print("\nDirectories:")
        print(f"Training Directory: {'✓' if validation['directories']['train']['exists'] else '✗'}")
        print(f"├── Image Count: {validation['directories']['train']['count']}")
        print(f"Test Directory: {'✓' if validation['directories']['test']['exists'] else '✗'}")
        print(f"├── Image Count: {validation['directories']['test']['count']}")

        print("\nLabel Files:")
        print(f"Training Labels: {'✓' if validation['labels']['train']['exists'] else '✗'}")
        print(f"Test Labels: {'✓' if validation['labels']['test']['exists'] else '✗'}")

    def setup(self) -> Tuple[bool, Dict]:
        """
        Complete dataset setup process.

        Returns:
            Tuple containing success status and validation results
        """
        # Mount drive
        if not self.mount_drive():
            return False, {}

        # Extract dataset
        if not self.extract_dataset():
            return False, {}

        # Validate dataset
        validation_results = self.validate_dataset()
        self.print_validation_results(validation_results)

        # Check if all components are present
        setup_success = (
            validation_results['directories']['train']['exists'] and
            validation_results['directories']['test']['exists'] and
            validation_results['labels']['train']['exists'] and
            validation_results['labels']['test']['exists'] and
            validation_results['directories']['train']['count'] > 0 and
            validation_results['directories']['test']['count'] > 0
        )

        if not setup_success:
            logger.warning("Dataset setup incomplete - some components are missing")
        else:
            logger.info("Dataset setup completed successfully")

        return setup_success, validation_results

# Execute the setup
loader = DatasetLoader()
success, validation = loader.setup()

# If successful, make paths easily accessible
if success:
    TRAIN_DIR = loader.paths['train_dir']
    TEST_DIR = loader.paths['test_dir']
    TRAIN_LABELS = loader.paths['train_labels']
    TEST_LABELS = loader.paths['test_labels']
    print("\nPaths are now ready for use in subsequent steps")
else:
    print("\nWARNING: Setup incomplete - please check the validation results above")


## Image Processing

class ImageProcessor:
    """Handles image validation and preprocessing."""

    @staticmethod
    def validate_and_resize_images(directory: str, target_size: Tuple[int, int]=(224, 224)) -> Dict:
        stats = {'total': 0, 'processed': 0, 'errors': 0, 'files': set()}

        # Look for images with different extensions
        extensions = ('*.jpg', '*.JPG')

        for ext in extensions:
            for img_path in tqdm(Path(directory).rglob(ext),
                               desc=f"Processing {ext} in {Path(directory).name}"):
                stats['total'] += 1
                try:
                    with Image.open(img_path) as img:
                        if img.size != target_size:
                            img = img.convert('RGB')
                            img = img.resize(target_size, Image.LANCZOS)
                            img.save(img_path, quality=95, format='JPEG')
                            stats['processed'] += 1
                        stats['files'].add(img_path.name)
                except Exception as e:
                    stats['errors'] += 1
                    logger.error(f"Error processing {img_path}: {str(e)}")

        print(f"\nDirectory {Path(directory).name}:")
        print(f"Total images: {stats['total']}")
        print(f"Processed: {stats['processed']}")
        print(f"Errors: {stats['errors']}")
        return stats

ImageProcessor.validate_and_resize_images(TRAIN_DIR)
ImageProcessor.validate_and_resize_images(TEST_DIR)


## Label Processing

class LabelProcessor:
    """Handles label validation and processing."""

    @staticmethod
    def process_labels(csv_path: str, img_dir: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df['full_path'] = df['file_path'].apply(
            lambda x: os.path.join(img_dir, Path(x).name)
        )
        valid_files = df['full_path'].apply(os.path.exists)
        valid_df = df[valid_files].copy()

        print(f"Label processing for {Path(csv_path).name}:")
        print(f"Total entries: {len(df)}")
        print(f"Valid entries: {len(valid_df)}")

        return valid_df

# Process labels
train_df = LabelProcessor.process_labels(TRAIN_LABELS, TRAIN_DIR)
test_df = LabelProcessor.process_labels(TEST_LABELS, TEST_DIR)

class LabelAnalyzer:
    """Analyzes and verifies label distribution in datasets."""

    @staticmethod
    def analyze_labels(dataframe: pd.DataFrame, name: str, expected_classes: int = 200) -> None:
        """
        Analyze label distribution in dataset.

        Args:
            dataframe: DataFrame containing labels
            name: Name of the dataset for reporting
            expected_classes: Expected number of unique classes
        """
        unique_labels = dataframe['Label'].unique()
        label_counts = dataframe['Label'].value_counts().sort_index()

        print(f"\n{name} Label Analysis:")
        print(f"Number of unique classes: {len(unique_labels)}")
        print(f"Label range: {min(unique_labels)} - {max(unique_labels)}")
        print(f"Missing classes: {set(range(expected_classes)) - set(unique_labels)}")

        print("\nClass distribution:")
        print(f"Min samples per class: {label_counts.min()}")
        print(f"Max samples per class: {label_counts.max()}")
        print(f"Mean samples per class: {label_counts.mean():.2f}")

        # Plot label distribution
        plt.figure(figsize=(15, 5))
        plt.bar(label_counts.index, label_counts.values)
        plt.title(f'{name} Label Distribution')
        plt.xlabel('Class Label')
        plt.ylabel('Number of Samples')
        plt.show()

# Label Processing with Verification
class LabelProcessor:
    """Handles label validation and processing."""

    @staticmethod
    def process_labels(csv_path: str, img_dir: str, expected_classes: int = 200) -> pd.DataFrame:
        """
        Process and validate labels with class verification.

        Args:
            csv_path: Path to labels CSV
            img_dir: Directory containing images
            expected_classes: Expected number of unique classes
        """
        df = pd.read_csv(csv_path)

        # Verify label range
        if not all(0 <= label < expected_classes for label in df['Label']):
            invalid_labels = df[~df['Label'].between(0, expected_classes-1)]
            print(f"\nWARNING: Found invalid labels in {Path(csv_path).name}:")
            print(invalid_labels[['file_path', 'Label']])

        # Process paths and validate existence
        df['full_path'] = df['file_path'].apply(
            lambda x: os.path.join(img_dir, Path(x).name)
        )
        valid_files = df['full_path'].apply(os.path.exists)
        valid_df = df[valid_files].copy()

        # Analyze label distribution
        LabelAnalyzer.analyze_labels(valid_df, Path(csv_path).stem)

        return valid_df

# Reprocess labels with verification
train_df = LabelProcessor.process_labels(TRAIN_LABELS, TRAIN_DIR)
test_df = LabelProcessor.process_labels(TEST_LABELS, TEST_DIR)


## Dataset Class

class BirdClassificationDataset(Dataset):
    """Custom Dataset for bird image classification."""

    def __init__(self, data: pd.DataFrame, transform: Optional[transforms.Compose] = None, shuffle: bool = True):
        """
        Initialize dataset with optional shuffling.

        Args:
            data: DataFrame containing image paths and labels
            transform: Optional transforms to be applied
            shuffle: Whether to shuffle the dataset indices
        """
        self.data = data.copy()  # Create a copy to avoid modifying original
        self.transform = transform

        # Shuffle data
        if shuffle:
            self.data = self.data.sample(frac=1, random_state=42).reset_index(drop=True)

        self.validate_labels()

    def validate_labels(self) -> None:
        """Validate label distribution in dataset."""
        labels = self.data['Label'].values
        unique_labels, counts = np.unique(labels, return_counts=True)

        print(f"\nDataset Label Validation:")
        print(f"Number of samples: {len(labels)}")
        print(f"Label range: {labels.min()} - {labels.max()}")
        print(f"Number of unique classes: {len(unique_labels)}")
        print(f"Samples per class - Min: {counts.min()}, Max: {counts.max()}, Mean: {counts.mean():.2f}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.data.iloc[idx]['full_path']
        label = self.data.iloc[idx]['Label']

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label


## Data Transformation

class DataTransformations:
    """Handles data augmentation and transformations."""

    @staticmethod
    def create_transforms() -> Dict[str, transforms.Compose]:
        """Create train and validation transforms."""
        transform_config = {
            'train': [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])
            ],
            'val': [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])
            ]
        }

        transforms_dict = {
            phase: transforms.Compose(config)
            for phase, config in transform_config.items()
        }

        # Print transform configurations
        print("\nTransform Configurations:")
        for phase, transform in transforms_dict.items():
            print(f"\n{phase} transforms:")
            for t in transform.transforms:
                print(f"- {t.__class__.__name__}")

        return transforms_dict

# Create and verify transforms
transforms_dict = DataTransformations.create_transforms()

def create_weighted_sampler(dataset: BirdClassificationDataset):
    """Create a weighted sampler to handle class imbalance."""
    labels = dataset.data['Label'].values
    class_counts = np.bincount(labels)
    weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = weights[labels]
    return WeightedRandomSampler(sample_weights.tolist(), len(sample_weights))


## Data Loaders

class DataLoaderBuilder:
    @staticmethod
    def create_loaders(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        transforms_dict: Dict[str, transforms.Compose],
        batch_size: int = 32
    ) -> Dict[str, DataLoader]:
        """Create data loaders with weighted sampling for training."""

        # Split training data
        train_df, val_df = train_test_split(
            train_df,
            test_size=0.2,
            stratify=train_df['Label'],
            random_state=42
        )

        # Create datasets
        datasets = {
            'train': BirdClassificationDataset(
                train_df,
                transforms_dict['train'],
                shuffle=True
            ),
            'val': BirdClassificationDataset(
                val_df,
                transforms_dict['val'],
                shuffle=True
            ),
            'test': BirdClassificationDataset(
                test_df.sample(frac=1, random_state=42).reset_index(drop=True),  # Shuffle test set
                transforms_dict['val'],
                shuffle=True
            )
        }

        # Create loaders with weighted sampling for training
        loaders = {}

        # Training loader with weighted sampling
        loaders['train'] = DataLoader(
            datasets['train'],
            batch_size=batch_size,
            sampler=create_weighted_sampler(datasets['train']),  # Use weighted sampler
            num_workers=2,
            pin_memory=True,
            drop_last=False
        )

        # Validation and test loaders with regular shuffling
        for phase in ['val', 'test']:
            loaders[phase] = DataLoader(
                datasets[phase],
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
                drop_last=False
            )

        return loaders
# Create and verify dataloaders
dataloaders = DataLoaderBuilder.create_loaders(train_df, test_df, transforms_dict)


## Data Preparation Validation

# Verify random access
def verify_random_access(dataloaders: Dict[str, DataLoader]) -> None:
    """Verify random access to samples across multiple iterations."""
    print("\nRandom Access Verification:")

    for phase, loader in dataloaders.items():
        print(f"\n{phase} dataset:")
        # Check first batch labels across multiple iterations
        labels_across_iterations = []
        for _ in range(3):
            _, labels = next(iter(loader))
            labels_across_iterations.append(sorted(labels.tolist()))
            print(f"Iteration labels: {labels_across_iterations[-1]}")

# Verify random access
verify_random_access(dataloaders)

def verify_dataloader(dataloader: DataLoader, phase: str) -> None:
    """Verify dataloader output format and shape."""
    images, labels = next(iter(dataloader))
    print(f"\n{phase} dataloader verification:")
    print(f"Batch image shape: {images.shape}")
    print(f"Batch label shape: {labels.shape}")
    print(f"Label range: {labels.min().item()} - {labels.max().item()}")

# Verify each dataloader
for phase, loader in dataloaders.items():
    verify_dataloader(loader, phase)


# **MODEL ARCHITECTURE**
class Config:
    num_classes = 200
    lr = 0.01
    weight_decay = 1e-4
    epochs = 50
    early_stopping_patience = 10
    scheduler_patience = 5
    scheduler_factor = 0.1
    label_smoothing = 0.2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dir = '/content/dataset/Dataset/Models'
    def __init__(self):
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)

config = Config()
print(f"Using device: {config.device}")
print(f"Model will be saved in: {config.model_dir}")


## EfficientNet Base Model Architecture
class BirdClassifier(nn.Module):
    def __init__(self, num_classes: int = 200):
        super(BirdClassifier, self).__init__()

        # Load base model
        self.efficientnet = timm.create_model(
            'efficientnet_b0',
            pretrained=True,
            num_classes=num_classes
        )

        # Unfreeze last 50 layers
        parameters = list(self.efficientnet.parameters())
        trainable_params = sum(p.numel() for p in parameters[-50:])
        total_params = sum(p.numel() for p in parameters)

        # Remove the default classifier
        in_features = self.efficientnet.classifier.in_features
        self.efficientnet.classifier = nn.Identity()

        # Add new classifier head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, num_classes)
        )

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.efficientnet(x)
        return self.classifier(x)

model = BirdClassifier(num_classes=config.num_classes)
model = model.to(config.device)


## Model Components

# Initialize model components
model = BirdClassifier(num_classes=config.num_classes)
model = model.to(config.device)

# Loss function with increased label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

# Change to SGD optimizer with momentum
optimizer = optim.SGD(
    model.parameters(),
    lr=config.lr,
    momentum=0.9,
    nesterov=True
)

# Keep the current scheduler
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    patience=config.scheduler_patience,
    factor=config.scheduler_factor,
    verbose=True
)
print("Training components initialized:")
print(f"Optimizer: Adam(lr={config.lr}, weight_decay={config.weight_decay})")
print(f"Loss function: CrossEntropyLoss with label_smoothing=0.1")
print(f"Scheduler: ReduceLROnPlateau(patience={config.scheduler_patience}, factor={config.scheduler_factor})")


# **MODEL TRAINING**
## Training and Evaluation Functions

@torch.no_grad()
def evaluate_model(model: nn.Module,
                  dataloader: torch.utils.data.DataLoader,
                  criterion: nn.Module,
                  device: torch.device) -> Tuple[float, float, Dict[int, float]]:
    """Evaluate model performance."""
    model.eval()
    running_loss = 0.0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    all_preds = []
    all_labels = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)

        running_loss += loss.item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        for pred, label in zip(preds, labels):
            class_total[label.item()] += 1
            if pred == label:
                class_correct[label.item()] += 1

    epoch_loss = running_loss / len(dataloader)
    top1_acc = np.mean(np.array(all_preds) == np.array(all_labels))

    class_accuracies = {
        class_idx: class_correct[class_idx] / class_total[class_idx]
        for class_idx in range(config.num_classes)
        if class_total[class_idx] > 0
    }

    return epoch_loss, top1_acc, class_accuracies

def train_epoch(model: nn.Module,
                dataloader: torch.utils.data.DataLoader,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    predictions = []
    targets = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = np.mean(np.array(predictions) == np.array(targets))

    return epoch_loss, epoch_acc

print("Training and evaluation functions initialized")


## Training Loop
history = defaultdict(list)
best_acc = 0.0
patience_counter = 0
start_time = time.time()

for epoch in range(config.epochs):
    print(f'\nEpoch {epoch+1}/{config.epochs}')
    print('-' * 10)

    # Train phase
    train_loss, train_acc = train_epoch(
        model, dataloaders['train'], criterion, optimizer, config.device
    )

    # Validation phase
    val_loss, val_acc, val_class_acc = evaluate_model(
        model, dataloaders['val'], criterion, config.device
    )

    # Save metrics
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_class_acc'].append(val_class_acc)

    # Update learning rate
    scheduler.step(val_acc)

    # Print statistics
    print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
    print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
    print(f'Average Class Acc: {np.mean(list(val_class_acc.values())):.4f}')

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(),
                  os.path.join(config.model_dir, 'best_model.pth'))
        print("Saved best model")
    else:
        patience_counter += 1

    if patience_counter >= config.early_stopping_patience:
        print('Early stopping triggered')
        break

time_elapsed = time.time() - start_time
print(f'\nTraining completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
print(f'Best val Acc: {best_acc:4f}')


## Training History
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Validation')
plt.title('Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train')
plt.plot(history['val_acc'], label='Validation')
plt.title('Accuracy History')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()



## **MODEL EVALUATION**
model.load_state_dict(torch.load(os.path.join(config.model_dir, 'best_model.pth')))

# Evaluate on test set
test_loss, test_acc, test_class_acc = evaluate_model(
    model, dataloaders['test'], criterion, config.device
)

print('\nTest Set Evaluation:')
print(f'Top-1 Accuracy: {test_acc:.4f}')
print(f'Average Class Accuracy: {np.mean(list(test_class_acc.values())):.4f}')
