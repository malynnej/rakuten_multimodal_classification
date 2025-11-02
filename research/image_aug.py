#Import for image augmentation and model training
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing import image
tf.config.list_physical_devices('GPU')

# # 3.0 Image Augmentation & Modeling @Jenny

# #Custom CNN model training with tf.data pipeline with Keras
# #Framework: TensorFlow 2.x with Keras
# #Using tf.data for efficient data loading and preprocessing augmentation
# #Using Keras preprocessing layers for data augmentation
# #Using a simple CNN architecture for demonstration
# #Using categorical crossentropy loss for multi-class classification
 # print("\nTrain model...")
AUTOTUNE = tf.data.AUTOTUNE

 # Create tf.data datasets from directory and create training and validation splits
train_ds = tf.keras.utils.image_dataset_from_directory(
     paths.Paths.image_train_path_output,
     labels='inferred',
     label_mode='categorical',
     image_size = (224, 224),  
     batch_size= 64,
     validation_split=0.2,
     subset='training',
     seed=42,
     shuffle=True
 )

 # Activate Limit to ~2000 samples (32 batches of 64)
 #train_ds = train_ds.take(32)

val_ds = tf.keras.utils.image_dataset_from_directory(
    paths.Paths.image_train_path_output,
    labels='inferred',
    label_mode='categorical',
    image_size = (224, 224), 
    batch_size = 64,
    validation_split=0.2,
    subset='validation',
    seed=42,
    shuffle=True
)

# Rescaling layer (ensure values are floats in [0,1])
rescale = tf.keras.layers.Rescaling(1.0 / 255)

# Optional: data augmentation as tf.keras layers (runs on GPU). Do NOT include rescaling here.
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
], name='data_augmentation')

# Best practice: rescale -> cache -> augment (train only) -> prefetch
train_ds = train_ds.map(lambda x, y: (rescale(x), y), num_parallel_calls=AUTOTUNE)
train_ds = train_ds.cache()  # cache the rescaled images (in memory or to disk if filename provided)
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
train_ds = train_ds.prefetch(AUTOTUNE)

# Validation: rescale -> cache -> prefetch (no augmentation)
val_ds = val_ds.map(lambda x, y: (rescale(x), y), num_parallel_calls=AUTOTUNE)
val_ds = val_ds.cache()
val_ds = val_ds.prefetch(AUTOTUNE)

# Quick sanity checks
print('Train classes:', train_ds.class_names if hasattr(train_ds, 'class_names') else 'n/a')
print('Train element spec:', train_ds.element_spec)
print('Validation element spec:', val_ds.element_spec)

# Now you can use train_ds and val_ds in model.fit
#history = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=[])