import os
import hashlib
import shutil
import json
from tqdm import tqdm

def get_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def merge_and_dedup(source_dirs, target_dir, class_name, existing_hashes):
    os.makedirs(target_dir, exist_ok=True)
    count = 0
    
    # Collect all valid files first to use with tqdm
    files_to_process = []
    for source_dir in source_dirs:
        if not os.path.exists(source_dir):
            continue
            
        for root, _, files in os.walk(source_dir):
            # Only process files if the parent folder matches the class name
            if os.path.basename(root) != class_name:
                continue
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    files_to_process.append(os.path.join(root, file))
                    
    for file_path in tqdm(files_to_process, desc=f"Merging {class_name}"):
        file_hash = get_md5(file_path)
        
        if file_hash not in existing_hashes:
            existing_hashes.add(file_hash)
            shutil.copy2(file_path, os.path.join(target_dir, os.path.basename(file_path)))
            count += 1
            
    return count

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    chest_xray_1 = os.path.join(data_dir, 'chest_xray_1')
    chest_xray_2 = os.path.join(data_dir, 'chest_xray_2')
    merged_dir = os.path.join(data_dir, 'merged')
    
    classes = ['NORMAL', 'PNEUMONIA']
    splits = ['train', 'val', 'test']
    
    stats = {}
    existing_hashes = set()
    
    for split in splits:
        stats[split] = {}
        for cls in classes:
            print(f"\nProcessing {split} - {cls}...")
            target_dir = os.path.join(merged_dir, split, cls)
            
            source_dirs = []
            if split == 'train':
                source_dirs = [os.path.join(chest_xray_1, split), os.path.join(chest_xray_2, split)]
            else:
                # Use chest_xray_1 val/test only
                source_dirs = [os.path.join(chest_xray_1, split)]
                
            count = merge_and_dedup(source_dirs, target_dir, cls, existing_hashes)
            stats[split][cls] = count
            
    # Calculate totals and imbalance for training set
    train_normal = stats['train']['NORMAL']
    train_pneumonia = stats['train']['PNEUMONIA']
    total_train = train_normal + train_pneumonia
    
    imbalance_ratio = train_pneumonia / train_normal if train_normal > 0 else 0
    
    stats['summary'] = {
        'total_train_normal': train_normal,
        'total_train_pneumonia': train_pneumonia,
        'train_imbalance_ratio_pneumonia_to_normal': imbalance_ratio
    }
    
    print("\n--- Final Counts (Train) ---")
    print(f"Total NORMAL: {train_normal}")
    print(f"Total PNEUMONIA: {train_pneumonia}")
    print(f"Imbalance Ratio (PNEUMONIA / NORMAL): {imbalance_ratio:.2f}")
    
    stats_file = os.path.join(merged_dir, 'dataset_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)
        
    print(f"\nSaved dataset stats to {stats_file}")

if __name__ == "__main__":
    main()
