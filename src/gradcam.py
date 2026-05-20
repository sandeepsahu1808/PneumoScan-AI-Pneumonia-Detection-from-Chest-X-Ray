import os
import numpy as np
import tensorflow as tf
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model

# Force CPU for evaluation to prevent Mac GPU (MPS) freezing
tf.config.set_visible_devices([], 'GPU')

def get_gradcam_heatmap(model, img_array, last_conv_layer_name='conv5_block3_out'):
    # In this architecture, ResNet50 layers and custom layers are flattened
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        tape.watch(last_conv_layer_output)
        
        # We are doing binary classification (1 output node)
        class_channel = preds[:, 0]

    # Gradient of the output class with respect to the output feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Vector where each entry is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map array by "how important this channel is"
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    # Handle NaN values which might occur if gradients are zero
    heatmap = tf.where(tf.math.is_nan(heatmap), tf.zeros_like(heatmap), heatmap)
    
    return heatmap.numpy()

def overlay_heatmap(heatmap, original_img, alpha=0.4):
    if original_img.max() <= 1.0:
        original_img = np.uint8(255 * original_img)
    else:
        original_img = np.uint8(original_img)

    h, w = original_img.shape[:2]
    heatmap_resized = np.array(
        Image.fromarray(np.uint8(255 * heatmap)).resize((w, h), Image.Resampling.BILINEAR)
    ) / 255.0

    heatmap_rgb = (cm.jet(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)
    superimposed_img = heatmap_rgb * alpha + original_img * (1 - alpha)
    return np.clip(superimposed_img, 0, 255).astype(np.uint8)

def generate_gradcam_samples(model, test_gen, n_samples=10, save_dir='results/gradcam/'):
    os.makedirs(save_dir, exist_ok=True)
    
    class_labels = {v: k for k, v in test_gen.class_indices.items()}
    
    # Get a batch
    test_gen.reset()
    images, labels = next(test_gen)
    
    # If batch size is smaller than n_samples, we might need multiple batches, but batch_size=32 is standard
    n_samples = min(n_samples, len(images))
    
    print(f"Generating {n_samples} Grad-CAM visualizations...")
    
    for i in range(n_samples):
        img_array = np.expand_dims(images[i], axis=0)
        true_class = int(labels[i])
        
        pred_prob = model.predict(img_array, verbose=0)[0][0]
        pred_class = 1 if pred_prob > 0.5 else 0
        
        heatmap = get_gradcam_heatmap(model, img_array)
        overlaid_img = overlay_heatmap(heatmap, images[i])
        
        # Plotting
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(images[i])
        plt.title(f"Original\nTrue: {class_labels[true_class]}")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(overlaid_img)
        color = 'green' if true_class == pred_class else 'red'
        plt.title(f"Grad-CAM\nPred: {class_labels[pred_class]} ({pred_prob:.2f})", color=color)
        plt.axis('off')
        
        status = "correct" if true_class == pred_class else "incorrect"
        filename = f"sample_{i+1}_{class_labels[true_class]}_pred_{class_labels[pred_class]}_{status}.png"
        plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight')
        plt.close()
        
    print(f"Saved visualizations to {save_dir}")

if __name__ == '__main__':
    from data_pipeline import get_generators

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    merged_data_dir = os.path.join(base_dir, 'data', 'merged')
    models_dir = os.path.join(base_dir, 'models')
    
    model_path = os.path.join(models_dir, 'best_model.h5')
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        exit()
        
    print("Loading model...")
    model = load_model(model_path, compile=False)
    
    print("Loading test data generator...")
    _, _, test_gen, _ = get_generators(merged_data_dir)
    
    generate_gradcam_samples(model, test_gen, n_samples=10)
