import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import cv2
from skimage import metrics

import os
import time


class ImageCleaning:
    def __init__(self, stats_calc=False, image_train_path="./data/raw/images/image_train/", df_save_path="./data/tmp/df_image.csv"):
        self.log = []
        self.stats_calc = stats_calc
        self.df_save_path = df_save_path
        self.image_train_path = image_train_path

        print("#" * 50, "\n")


        print("\n")
        print("#" * 50)


    def image_stats_calc(self, dataframe):
        t_start = time.time()

        df = dataframe.copy()

        gray_means, gray_stds, red_means, red_stds, green_means, green_stds, blue_means ,blue_stds, image_filenames = [], [], [], [], [], [], [], [], []

        for image_name in os.listdir(self.image_train_path):
            image_filenames.append(image_name)

            image_gray = cv2.imread(self.image_train_path+image_name, cv2.IMREAD_GRAYSCALE)
            gray_means.append(np.mean(image_gray))
            gray_stds.append(np.std(image_gray))

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
        
        df.to_csv(self.df_save_path+"df_image.csv")

        t_end = time.time()
        t_exec = t_end-t_start
        print(f"Execution time of image_stats_calc(): {t_exec}")

        return df

    def remove_blanks(self, dataframe):

        df = dataframe.copy()

        pass
    
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

                    if ssim_score > 0.90:
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

    def remove_duplicates(self, dataframe):

        unique_means = dataframe["image_gray_mean"].unique()

        ctr = 0
        similarities = {}
        path = "./data/raw/images/image_train/"

        for mean in unique_means:
            ctr += 1
            print(f"{ctr} / {len(unique_means)}")
            
            duplicated_images = dataframe[dataframe["image_gray_mean"] == mean]

            update = self.structural_similarity(duplicated_images, path)
            
            if update:
                similarities.update(update)

        duplicates = set([element for entry in similarities.values() for element in entry["duplicates"]])

        return duplicates
        
    
    def remove_noise(self, dataframe):
        pass

    def clean_images(self, dataframe):

        if self.stats_calc:
            df = self.image_stats_calc(df)

        pass

            

