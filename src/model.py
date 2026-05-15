import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def build_model(img_size=224, trainable_base=False):
    base_model = ResNet50(
        weights='imagenet', 
        include_top=False, 
        input_shape=(img_size, img_size, 3)
    )
    
    if not trainable_base:
        for layer in base_model.layers:
            layer.trainable = False
            
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model

def unfreeze_model(model, num_layers=30):
    for layer in model.layers[-num_layers:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True
            
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model

def model_summary_to_file(model, path='results/model_summary.txt'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

if __name__ == '__main__':
    print("Building model...")
    model = build_model()
    
    print("Saving model summary to results/model_summary.txt")
    model_summary_to_file(model)
    print("Done!")
