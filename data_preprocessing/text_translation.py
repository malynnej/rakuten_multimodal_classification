###########################################################################################
#
# Main Data Dreprocessing Pipeline
#
# 1. Textual Data Preprocessing
#
# 1.1 
#   Text Cleaning - Converting the text from html to normal text
#   Fixing Some encoded words into normal ones

# 1.2 Text Translation
#    Text Translation - using Google translation or API
#
###########################################################################################


# Import Necessary libraries 

# ----------------------------------
# 📦 Data & Numerical Analysis
# ----------------------------------

import pandas as pd
import numpy as np
# ----------------------------------
# 🌏  Translation tasks
# ----------------------------------

from deep_translator import GoogleTranslator
from tqdm import tqdm
tqdm.pandas()
import ollama

class LLMTransformation:

    """
    The class is perform the translation tasks of the DataFrame. 

    Parameters :
    --------------------------
    
        Kind of translator : GoogleAPI or LLM etc.
    
    """

    def __init__(self, TranslatorKind='LLM', llm_model="llama3.2:3b"):

        self.TranslatorKind=TranslatorKind

        if self.TranslatorKind == "LLM":
            self.model = llm_model
            result = ollama.generate(model=self.model, prompt='Just write "Ollam is working"')
            print("*" * 50)
            print(f"-- The ollama LLM model,{self.model} is being used.")
            print("--",result['response'])                
        else:
            print("*" * 50)
            print(f"-- Translator kind is {self.TranslatorKind}")

        print("-- Translator object initialized successfully.")
        print("*"* 50)

    # -----------------------------------------------------
    # --------------- Select translate function -----------
    # -----------------------------------------------------

    def TxtTranslate(self, df, columns):

        df = df.copy()

        if self.TranslatorKind == 'LLM':
            df = self.translateByLLM(df, columns)

        if self.TranslatorKind == 'Google': 
            df = self.translateByGoogle(df, columns)

        return df
    
    # ----------- LLM translation function -------------

    def translateByLLM(self, df, columns):

        prompt = input("Enter your prompt:\n ")
        print(prompt)

        def ollama_translate(inText):
            full_prompt = prompt + f"Input Text: {inText}"
            response = ollama.chat(model=self.model, messages=[{"role":"user","content":full_prompt}])
            return response["message"]["content"].strip()
        
        df = df.copy()

        for col in columns:
            if col not in df.columns:
                print(f"Column '{col}' not found in DataFrame. Skipping.")
                continue

            OutCol=f"{col}_en"
            df[OutCol] = df.progress_apply(
                lambda row: ollama_translate(row[col]), axis=1)

        return df
    
    # ----------- Google translation -----

    def translateByGoogle(self, df, columns):
        
        error_count = 0
        # Fetch text and translate
        def _Gtranslate(text):
            nonlocal error_count
            if pd.isnull(text) or text.strip() == "":
                return text
            try:
                translated = GoogleTranslator(source='auto', target='en').translate(text)
                return translated
        
            except Exception as e:
                print(f"Error translating '{text[:30]}...': {e}")
                error_count += 1
                return text  # Return original text if translation fails
            
        df = df.copy()

        for col in columns:
            if col not in df.columns:
                print(f"Column '{col}' not found in DataFrame. Skipping.")
                continue

            OutCol=f"{col}_en"
            df["OutCol"] = df[col].progress_apply(_Gtranslate)

        print(f"\n Total sentence more than 5000 character, {error_count}")
        return df


    