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




# 2.2 Image Augmentation @Jenny




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