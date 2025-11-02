import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import cv2
from skimage import metrics

from tqdm import tqdm
import os
import time


class ImageCleaning:
    def __init__(self, stats_calc=False, color_stats=False, ssim_threshold=0.95, image_train_path="./data/raw/images/image_train/", df_save_path="./data/tmp/df_image.csv", overwrite_duplicates=False, duplicates_path="./data/tmp/duplicates.csv"):
        self.log = []
        self.stats_calc = stats_calc
        self.color_stats = color_stats
        self.ssim_threshold = ssim_threshold
        self.df_save_path = df_save_path
        self.image_train_path = image_train_path
        self.overwrite_duplicates = overwrite_duplicates
        self.duplicates_path = duplicates_path

        
        print("#" * 50, "\n")
        print("Image Cleaning:\n")
        print("#" * 50, "\n")

        print("Parameter:")
        print("-" * 50)
        print(f"stats_calc:             {self.stats_calc}")
        print(f"overwrite_duplicates:   {self.overwrite_duplicates}")
        print(f"ssim_threshold:         {self.ssim_threshold}")

        print("\nPaths:")
        print("-" * 50)
        print(f"df_save_path:           {self.df_save_path}")
        print(f"image_train_path:       {self.image_train_path}")
        print(f"duplicates_path:        {self.duplicates_path}\n")
        print("#" * 50, "\n")


    def image_stats_calc(self, dataframe):
        t_start = time.time()

        df = dataframe.copy()

        gray_means, gray_stds, red_means, red_stds, green_means, green_stds, blue_means ,blue_stds, image_filenames = [], [], [], [], [], [], [], [], []

        for image_name in tqdm(df["image_name"], desc="Calculate statistics"):
            image_filenames.append(image_name)

            image_gray = cv2.imread(self.image_train_path+image_name, cv2.IMREAD_GRAYSCALE)
            gray_means.append(np.mean(image_gray))
            gray_stds.append(np.std(image_gray))

            if self.color_stats:
                image_color = cv2.imread(self.image_train_path+image_name, cv2.IMREAD_COLOR)
                red_means.append(np.mean(image_color[:,:,0]))
                red_stds.append(np.std(image_color[:,:,0]))

                green_means.append(np.mean(image_color[:,:,1]))
                green_stds.append(np.std(image_color[:,:,1]))

                blue_means.append(np.mean(image_color[:,:,2]))
                blue_stds.append(np.std(image_color[:,:,2]))

        df["image_gray_mean"] = gray_means
        df["image_gray_std"] = gray_stds
        df["image_gray_var"] = df["image_gray_std"] * df["image_gray_std"]

        if self.color_stats:
            df["image_red_mean"] = red_means
            df["image_red_std"] = red_stds
            df["image_red_var"] = df["image_red_std"] * df["image_red_std"]

            df["image_green_mean"] = green_means
            df["image_green_std"] = green_stds
            df["image_green_var"] = df["image_green_std"] * df["image_green_std"]

            df["image_blue_mean"] = blue_means
            df["image_blue_std"] = blue_stds
            df["image_blue_var"] = df["image_blue_std"] * df["image_blue_std"]
            df["image_name"] = image_filenames

        if not os.path.exists(self.df_save_path) or not os.path.isdir(self.df_save_path):
            os.makedirs(self.df_save_path)
        
        df.to_csv(self.df_save_path+"image_stats.csv")

        t_end = time.time()
        t_exec = t_end-t_start

        print("=" * 50)
        print(f"Execution time of image_stats_calc(): {t_exec}")
        print("=" * 50)

        return df

    def remove_blanks(self, dataframe):

        df = dataframe.copy()
        len_init = len(df)

        df = df[df["image_gray_var"] != 0].reset_index(drop=True)
        len_post = len(df)

        print("=" * 50)
        print(f"Blanks removed: {len_init-len_post}")
        print("=" * 50)

        return df
    
    def structural_similarity(self, df, path):

        similarities = {}
        processed = []

        for image_name in df["image_name"]:

            duplicates = []
            ssim = []
            mse = []

            processed.append(image_name)

            image = cv2.imread(path+image_name, cv2.IMREAD_GRAYSCALE)

            for image_comp_name in df.loc[df["image_name"] != image_name, "image_name"]:

                if image_comp_name not in processed:
                    image_comp = cv2.imread(path+image_comp_name, cv2.IMREAD_GRAYSCALE)
                    ssim_score = metrics.structural_similarity(image, image_comp, data_range=255)
                    mse_calc = metrics.mean_squared_error(image, image_comp)

                    if ssim_score > self.ssim_threshold:
                        duplicates.append(image_comp_name)
                        ssim.append(ssim_score)
                        mse.append(mse_calc)

            if duplicates:
                similarities.update({image_name: {
                                        "duplicates": duplicates,
                                        "SSIM": ssim,
                                        "MSE": mse
                                        }
                                    })

        return similarities

    def identify_duplicates(self, dataframe):

        unique_means = dataframe["image_gray_mean"].unique()

        similarities = {}

        for mean in tqdm(unique_means, desc="Identify duplicates"):
            
            duplicated_images = dataframe[dataframe["image_gray_mean"] == mean]

            update = self.structural_similarity(duplicated_images, self.image_train_path)
            
            if update:
                similarities.update(update)

        duplicates = set([element for entry in similarities.values() for element in entry["duplicates"]])

        image_similarites = pd.DataFrame.from_dict(similarities, orient="index").reset_index().rename(columns={"index": "image"})
        image_similarites.to_csv(self.df_save_path+"image_similarities.csv")

        print("=" * 50)
        print(f"Duplicates found: {len(duplicates)}")
        print("=" * 50)

        return duplicates
        
    
    def remove_noise(self, dataframe):
        pass


    def clean_images(self, dataframe):

        t_start = time.time()
 
        print(">>> Calculate image statistics.")
        if self.stats_calc:
            df = self.image_stats_calc(dataframe)
        else:
            df = pd.read_csv(self.df_save_path+"image_stats.csv")

        print(">>> Remove blank images.")
        df = self.remove_blanks(df)
        
        print(">>> Identify duplicates.")
        if self.overwrite_duplicates:
            duplicates = self.identify_duplicates(df)
        else:
            duplicates = pd.read_csv(self.duplicates_path)
        
        # Remove duplicates
        print(">>> Remove duplicates.")
        len_init = len(df)

        df = df[~df["image_name"].isin(duplicates)].reset_index(drop=True)

        len_post = len(df)
        print("=" * 50)
        print(f"Duplicates removed: {len_init-len_post}")
        print("=" * 50)

        t_end = time.time()
        t_exec = t_end-t_start

        print(f"Execution time of clean_images(): {t_exec}")

        return df

            

