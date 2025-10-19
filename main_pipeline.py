###########################################################################################
#
# Main Data Dreprocessing Pipeline
#
# 1. Textual Data Preprocessing
#
# 1.1 Text Cleaning @Rohan
# 1.2 Text Translation @Rohan
# 1.3 Text outliers @Michael
# 1.4 Balancing Textual Data @Johann
#
# 2. Image Data Preprocessing
#
# 2.1 Clean Image Data @Michael
# 2.2 Image Augmentation @Jenny
#
###########################################################################################

# Import libraries
import pandas as pd
import numpy as np


# Import custom classes for data preprocessing
from data_preprocessing.initialize_data import initialization
from data_preprocessing.text_outliers import TransformTextOutliers
from data_preprocessing.image_cleaning import ImageCleaning
from data_preprocessing.image_preprocessing import ImagePreprocessor, BackgroundDetector, ImageCropper

#Import for image augmentation and model training
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing import image
tf.config.list_physical_devices('GPU')

# Loading Data Frame
init = initialization()
df, paths, params = init.initialize()


# 1. Textual Data Preprocessing
# 1.1 Text Cleaning @Rohan
 
# 1.2 Text Translation @Rohan

# 1.3 Text outliers @Michael
tto = TransformTextOutliers(model=params.TextOutlier.llm,
                            sentence_normalization=params.TextOutlier.sentence_normalization,
                            similarity_threshold=params.TextOutlier.similarity_threshold, 
                            factor=params.TextOutlier.factor)

df, _, _ = tto.transform_outliers(df)

# 1.4 Balancing Textual Data @Johann

# 2. Image Data Preprocessing
# 2.1 Clean Image Data @Michael
IC = ImageCleaning(stats_calc=params.ImageCleaning.stats_calc,
                   image_train_path=paths.Paths.image_train_path,
                    df_save_path=paths.Paths.ImageDataFrameSavePath)
IC.image_stats_calc(df)


# 2.2 Image Preprocessing (Background Handling) @Jenny

print("Start image preprocessing...")

# Initialize components
detector = BackgroundDetector(
    white_threshold= params.BackgroundDetector.white_threshold,
    border_ratio=params.BackgroundDetector.border_ratio,
    white_percentage_threshold=params.BackgroundDetector.white_percentage_threshold,
)

cropper = ImageCropper(padding=params.ImageCropper.padding, min_crop_ratio=params.ImageCropper.min_crop_ratio, min_aspect_ratio=params.ImageCropper.min_aspect_ratio)

preprocessor = ImagePreprocessor(
    background_detector=detector,
    image_cropper=cropper
)

# Process dataset: images are in a flat folder; derive categories from dataframe 'prdtypecode'
preprocessor.preprocess_from_dataframe(
    df=df,
    output_dir=paths.Paths.image_train_path_output,
    output_dir_greyscale=paths.Paths.image_train_path_greyscale,
    image_path_col='image_path',
    category_col='prdtypecode'
)

# 3.0 Image Augmentation & Modeling @Jenny

#Custom CNN model training with tf.data pipeline with Keras
#Framework: TensorFlow 2.x with Keras
#Using tf.data for efficient data loading and preprocessing augmentation
#Using Keras preprocessing layers for data augmentation
#Using a simple CNN architecture for demonstration
#Using categorical crossentropy loss for multi-class classification
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


###########################################################################################

# Results

###########################################################################################

# Expected output:
# .csv-files:
# X_train, X_test, y_train, y_test
#
# Folder structure:
# Images
## Images/train/
### Images/train/prdtypecode/
#### image1.jpg
#### image2.jpg etc.
#
# Images
## Images/test/
### Images/test/prdtypecode/
#### image1.jpg
#### image2.jpg etc.