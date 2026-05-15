import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

from data_pipeline import get_generators
from model import build_model, unfreeze_model

def plot_training_history(history1, history2=None):
    acc1 = history1.history['accuracy']
    val_acc1 = history1.history['val_accuracy']
    loss1 = history1.history['loss']
    val_loss1 = history1.history['val_loss']
    
    if history2:
        acc2 = history2.history['accuracy']
        val_acc2 = history2.history['val_accuracy']
        loss2 = history2.history['loss']
        val_loss2 = history2.history['val_loss']
        
        acc = acc1 + acc2
        val_acc = val_acc1 + val_acc2
        loss = loss1 + loss2
        val_loss = val_loss1 + val_loss2
    else:
        acc = acc1
        val_acc = val_acc1
        loss = loss1
        val_loss = val_loss1
        
    epochs = range(1, len(acc) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    if history2:
        plt.axvline(x=len(acc1), color='k', linestyle='--', label='Fine-tuning starts')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    if history2:
        plt.axvline(x=len(loss1), color='k', linestyle='--', label='Fine-tuning starts')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/training_history.png')
    plt.close()

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    merged_data_dir = os.path.join(base_dir, 'data', 'merged')
    models_dir = os.path.join(base_dir, 'models')
    results_dir = os.path.join(base_dir, 'results')
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print("Loading data generators...")
    train_gen, val_gen, test_gen, class_weights = get_generators(merged_data_dir)
    
    # PHASE 1: Frozen base training
    print("\n=== PHASE 1: Frozen Base Training ===")
    model = build_model(trainable_base=False)
    
    callbacks_phase1 = [
        ModelCheckpoint(os.path.join(models_dir, 'best_model_phase1.h5'), 
                        monitor='val_recall', save_best_only=True, mode='max', verbose=1),
        EarlyStopping(monitor='val_recall', patience=5, restore_best_weights=True, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        CSVLogger(os.path.join(results_dir, 'training_log_phase1.csv'))
    ]
    
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=15,
        class_weight=class_weights,
        callbacks=callbacks_phase1
    )
    
    val_acc1 = history1.history.get('val_accuracy', [-1])[-1]
    val_auc1 = history1.history.get('val_auc', [-1])[-1]
    print(f"Phase 1 Final - Val Accuracy: {val_acc1:.4f}, Val AUC: {val_auc1:.4f}")
    
    # PHASE 2: Fine-tuning
    print("\n=== PHASE 2: Fine-Tuning ===")
    print("Loading best model from Phase 1...")
    model = load_model(os.path.join(models_dir, 'best_model_phase1.h5'))
    
    print("Unfreezing last 30 layers...")
    model = unfreeze_model(model, num_layers=30)
    
    callbacks_phase2 = [
        ModelCheckpoint(os.path.join(models_dir, 'best_model.h5'), 
                        monitor='val_recall', save_best_only=True, mode='max', verbose=1),
        EarlyStopping(monitor='val_recall', patience=5, restore_best_weights=True, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        CSVLogger(os.path.join(results_dir, 'training_log_phase2.csv'))
    ]
    
    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        class_weight=class_weights,
        callbacks=callbacks_phase2
    )
    
    val_acc2 = history2.history.get('val_accuracy', [-1])[-1]
    val_auc2 = history2.history.get('val_auc', [-1])[-1]
    print(f"Phase 2 Final - Val Accuracy: {val_acc2:.4f}, Val AUC: {val_auc2:.4f}")
    
    print("\nSaving final model...")
    model.save(os.path.join(models_dir, 'best_model.h5'))
    
    print("Plotting training history...")
    plot_training_history(history1, history2)
    print("Training pipeline complete! History saved to results/training_history.png")

if __name__ == '__main__':
    main()
