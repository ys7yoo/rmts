import os
import pandas as pd
import requests
from typing import Dict

def download_asap_dataset(save_dir: str = 'data/asap'):
    """
    Download and prepare ASAP dataset
    
    Args:
        save_dir (str): Directory to save dataset
    """
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    for fold in range(5):
        os.makedirs(os.path.join(save_dir, f'fold_{fold}'), exist_ok=True)

    # Kaggle dataset URLs (these are placeholder URLs, replace with actual links)
    dataset_urls = {
        'training': 'https://example.com/training_set.csv',
        'validation': 'https://example.com/validation_set.csv'
    }

    # Download datasets
    datasets = {}
    for name, url in dataset_urls.items():
        try:
            response = requests.get(url)
            response.raise_for_status()
            datasets[name] = pd.read_csv(pd.compat.StringIO(response.text))
        except Exception as e:
            print(f"Error downloading {name} dataset: {e}")
            return None

    # Preprocess and split into folds
    def prepare_fold_data(data: pd.DataFrame, fold: int):
        """Prepare data for a specific fold"""
        fold_dir = os.path.join(save_dir, f'fold_{fold}')
        
        # Split data into train, dev, test
        train_data = data.iloc[:-100]  # Most data for training
        dev_data = data.iloc[-100:-50]  # Some for validation
        test_data = data.iloc[-50:]    # Some for testing
        
        # Save to CSV
        train_data.to_csv(os.path.join(fold_dir, 'train.csv'), index=False)
        dev_data.to_csv(os.path.join(fold_dir, 'dev.csv'), index=False)
        test_data.to_csv(os.path.join(fold_dir, 'test.csv'), index=False)

    # Create 5 folds
    for fold in range(5):
        # Simple data shuffling for creating different folds
        shuffled_data = datasets['training'].sample(frac=1, random_state=fold)
        prepare_fold_data(shuffled_data, fold)

    print("ASAP Dataset preparation completed!")

def validate_dataset(save_dir: str = 'data/asap'):
    """
    Validate the created dataset structure
    
    Args:
        save_dir (str): Directory of saved dataset
    
    Returns:
        Dict with dataset validation information
    """
    validation_results: Dict[str, bool] = {}
    
    for fold in range(5):
        fold_path = os.path.join(save_dir, f'fold_{fold}')
        
        # Check if all required files exist
        validation_results[f'fold_{fold}'] = all([
            os.path.exists(os.path.join(fold_path, f'{split}.csv')) 
            for split in ['train', 'dev', 'test']
        ])
    
    return validation_results

if __name__ == "__main__":
    download_asap_dataset()
    results = validate_dataset()
    print("Dataset Validation Results:", results)
