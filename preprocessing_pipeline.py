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
import os
import time
from datetime import timedelta


# Import custom classes for data preprocessing
from data_preprocessing.initialize_data import initialization
from data_preprocessing.text_cleaning import TextCleaning
from data_preprocessing.text_translation import LLMTransformation
from data_preprocessing.text_outliers import TransformTextOutliers
from data_preprocessing.image_cleaning import ImageCleaning
from data_preprocessing.class_balancing import ClassBalancing # see 1.4 Balancing Textual Data @Johann
from data_preprocessing.image_preprocessing import ImagePreprocessor, BackgroundDetector, ImageCropper

# Loading Data Frame
init = initialization(dev_mode=True)
df_train, df_test, paths, params = init.initialize()

def preprocessing_pipeline(df, paths, params, mode):

    ###########################################################################################

    # 1. Textual Data Preprocessing

    ###########################################################################################

    # 1.1 Text Cleaning @Rohan
    if params.ExecFlags.TextCleaningFlag:
        clt = TextCleaning(html_to_text = params.DecodeEncode.html_to_text,
                        words_encoding = params.DecodeEncode.words_encoding)

        df,_, = clt.cleanTxt(df,["text"])

    # 1.2 Text outliers @Michael
    if params.ExecFlags.TransformTextOutliersFlag:
        tto = TransformTextOutliers(column_to_transform="text",
                                    model=params.TextOutlier.llm,
                                    word_count_threshold = params.TextOutlier.word_count_threshold,
                                    sentence_normalization=params.TextOutlier.sentence_normalization,
                                    similarity_threshold=params.TextOutlier.similarity_threshold, 
                                    factor=params.TextOutlier.factor)

        df, _, _ = tto.transform_outliers(df)
    
    # 1.3 Text Translation @Rohan
    # Instantiate the translation class
    if params.ExecFlags.LLMTransformationFlag:
        ttr = LLMTransformation(TranslatorKind=params.transPara.TranslatorKind, 
                            llm_model=params.transPara.llm_model,
                            batch_mode = params.transPara.batch_mode,
                            chunk_size= params.transPara.chunk_size
                            )

        # Short DataFrame for testing, taking long computation time
        # for big dataframe. Batch needs to be implemented. 
        df = ttr.TxtTranslate(df,["text"])

    # # 1.4 Balancing Textual Data @Johann
    if params.ExecFlags.ClassBalancingFlag and mode=="train":
        cb = ClassBalancing(method=params.ClassBalancing.method)
        df = cb.process(df)


    init.save_files(df, paths, mode, X_columns=["text"])

    ###########################################################################################

    # 2. Image Data Preprocessing

    ###########################################################################################

    # 2.1 Clean Image Data @Michael
    if params.ExecFlags.ImageCleaningFlag:
        IC = ImageCleaning(stats_calc=params.ImageCleaning.stats_calc,
                            color_stats=params.ImageCleaning.color_stats,
                            image_train_path=paths.Paths.image_train_path,
                            df_save_path=paths.Paths.ImageDataFrameSavePath,
                            overwrite_duplicates=params.ImageCleaning.overwrite_duplicates)
        
        df = IC.clean_images(df)


    # 2.2 Image Preprocessing (Background Handling) @Jenny
    if params.ExecFlags.ImagePreprocessorFlag:
        print("Start image preprocessing...")

        # Initialize components
        detector = BackgroundDetector(
            white_threshold= params.BackgroundDetector.white_threshold,
            border_ratio=params.BackgroundDetector.border_ratio,
            white_percentage_threshold=params.BackgroundDetector.white_percentage_threshold,
        )

        cropper = ImageCropper(
            padding=params.ImageCropper.padding,
            min_crop_ratio=params.ImageCropper.min_crop_ratio,
            min_aspect_ratio=params.ImageCropper.min_aspect_ratio)

        preprocessor = ImagePreprocessor(
            background_detector=detector,
            image_cropper=cropper
        )

        # Define paths according to mode
        if mode == "train":
            output_dir = paths.Paths.image_train_path_output
        elif mode == "test":
            output_dir = paths.Paths.image_test_path_output

        # Process dataset: images are in a flat folder; derive categories from dataframe 'prdtypecode'
        preprocessor.preprocess_from_dataframe(
            df=df,
            output_dir=output_dir,
            image_path_col='image_path',
            category_col='prdtypecode'
        )

# Start execution timer
t_start = time.time()

print(">>> Execute preprocessing pipeline for training data.")
preprocessing_pipeline(df_train, paths, params, mode="train")

print(">>> Execute preprocessing pipeline for test data.")
preprocessing_pipeline(df_test, paths, params, mode="test")

# End execution timer
t_end = time.time()
t_exec = str(timedelta(seconds=t_end-t_start))

print("\nData preprocessing done!")
print(f"Execution time: {t_exec}.")


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