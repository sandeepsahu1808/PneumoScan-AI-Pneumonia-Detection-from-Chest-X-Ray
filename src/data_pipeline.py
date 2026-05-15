import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
import numpy as np

def compute_class_weights(train_gen):
    classes = train_gen.classes
    class_weights_array = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(classes),
        y=classes
    )
    return {0: class_weights_array[0], 1: class_weights_array[1]}

def get_generators(data_dir, img_size=224, batch_size=32):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        shear_range=0.1,
        fill_mode='nearest'
    )
    
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary'
    )
    
    val_gen = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    test_gen = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    class_weights = compute_class_weights(train_gen)
    
    return train_gen, val_gen, test_gen, class_weights

def visualize_samples(train_gen, n=9):
    images, labels = next(train_gen)
    
    plt.figure(figsize=(10, 10))
    for i in range(min(n, len(images))):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        label = 'PNEUMONIA' if labels[i] == 1 else 'NORMAL'
        plt.title(label)
        plt.axis("off")
        
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/sample_augmentation.png')
    plt.close()

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    merged_data_dir = os.path.join(base_dir, 'data', 'merged')
    
    if os.path.exists(merged_data_dir):
        print(f"Loading data from {merged_data_dir}")
        train_gen, val_gen, test_gen, class_weights = get_generators(merged_data_dir)
        
        print("\nGenerator Stats:")
        print(f"Train samples: {train_gen.samples}")
        print(f"Validation samples: {val_gen.samples}")
        print(f"Test samples: {test_gen.samples}")
        print(f"Class weights: {class_weights}")
        
        print("\nVisualizing samples...")
        visualize_samples(train_gen)
        print("Saved sample visualization to results/sample_augmentation.png")
    else:
        print(f"Error: Merged data directory not found at {merged_data_dir}")
        print("Please ensure you have placed the datasets in data/chest_xray_1 and data/chest_xray_2 and run merge_datasets.py first.")
