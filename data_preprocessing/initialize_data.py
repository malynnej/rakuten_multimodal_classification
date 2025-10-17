import pandas as pd
import os

import yaml
from box import ConfigBox

class initialization:
    def __init__(self, config_path = "./config.yaml", params_path = "./params.yaml"):
        self.config_path = config_path
        self.params_path = params_path

    def read_yaml(self, path):
        with open(path) as yaml_file:
            content = yaml.safe_load(yaml_file)

            return ConfigBox(content)

    def load_data(self, paths):

        X_train = pd.read_csv(paths.Paths.X_train, index_col=0)
        y_train = pd.read_csv(paths.Paths.y_train, index_col=0)

        df = X_train.join(y_train)

        df["text"] = df["designation"] + " " + df["description"]
        df["text"] = df["text"].fillna(df["designation"])

        df = df.drop(columns=["designation", "description"])

        print("Data loaded successfully.")

        return df
    
    def initialize(self):
        paths = self.read_yaml(self.config_path)
        params = self.read_yaml(self.params_path)

        df = self.load_data(paths)

        return df, paths, params