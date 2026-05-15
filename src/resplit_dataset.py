import os
import shutil
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    merged_dir = os.path.join(base_dir, 'data', 'merged')
    
    classes = ['NORMAL', 'PNEUMONIA']
    splits = ['train', 'val', 'test']
    
    # Collect all image paths per class
    images = {'NORMAL': [], 'PNEUMONIA': []}
    
    for split in splits:
        for cls in classes:
            cls_dir = os.path.join(merged_dir, split, cls)
            if not os.path.exists(cls_dir):
                continue
            for file in os.listdir(cls_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    images[cls].append(os.path.join(cls_dir, file))
                    
    # Temp dir to hold files before deleting old folders
    temp_dir = os.path.join(base_dir, 'data', 'temp_merged')
    os.makedirs(temp_dir, exist_ok=True)
    
    new_splits = {'train': {'NORMAL': [], 'PNEUMONIA': []},
                  'val': {'NORMAL': [], 'PNEUMONIA': []},
                  'test': {'NORMAL': [], 'PNEUMONIA': []}}
                  
    for cls in classes:
        cls_images = images[cls]
        labels = [cls] * len(cls_images)
        
        # Split into 70% train, 30% temp (for val + test)
        train_img, temp_img, train_lbl, temp_lbl = train_test_split(
            cls_images, labels, test_size=0.30, random_state=42, stratify=labels
        )
        
        # Split temp into 50% val, 50% test (which equals 15% / 15% of total)
        val_img, test_img = train_test_split(
            temp_img, test_size=0.50, random_state=42, stratify=temp_lbl
        )
        
        new_splits['train'][cls] = train_img
        new_splits['val'][cls] = val_img
        new_splits['test'][cls] = test_img

    # Move to a temporary folder to avoid in-place deletion issues
    print("Moving files to temporary restructuring directory...")
    moves_planned = []
    
    for split in splits:
        for cls in classes:
            dest_dir = os.path.join(merged_dir, split, cls)
            for img_path in new_splits[split][cls]:
                filename = os.path.basename(img_path)
                temp_path = os.path.join(temp_dir, f"{split}_{cls}_{filename}")
                shutil.move(img_path, temp_path)
                moves_planned.append((temp_path, os.path.join(dest_dir, filename)))
                
    # Now that files are safely in temp, clear the old directories
    for split in splits:
        for cls in classes:
            d = os.path.join(merged_dir, split, cls)
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)
            
    # Move files from temp to final destinations
    for src, dst in tqdm(moves_planned, desc="Resplitting to final folders"):
        shutil.move(src, dst)
        
    shutil.rmtree(temp_dir)
    
    stats = {}
    for split in splits:
        stats[split] = {}
        for cls in classes:
            stats[split][cls] = len(new_splits[split][cls])
            
    # Print final counts
    print("\n--- Final Resplit Counts ---")
    for split in splits:
        print(f"- {split.capitalize()}: NORMAL {stats[split]['NORMAL']}, PNEUMONIA {stats[split]['PNEUMONIA']}, Total {stats[split]['NORMAL'] + stats[split]['PNEUMONIA']}")
        
    train_normal = stats['train']['NORMAL']
    train_pneumonia = stats['train']['PNEUMONIA']
    imbalance_ratio = train_pneumonia / train_normal if train_normal > 0 else 0
    
    stats['summary'] = {
        'total_train_normal': train_normal,
        'total_train_pneumonia': train_pneumonia,
        'train_imbalance_ratio_pneumonia_to_normal': imbalance_ratio
    }
    
    stats_file = os.path.join(merged_dir, 'dataset_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)
        
    print(f"\nSaved updated dataset stats to {stats_file}")

if __name__ == "__main__":
    main()
