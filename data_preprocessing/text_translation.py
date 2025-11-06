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
import time
import re

class LLMTransformation:

    """
    The class is perform the translation tasks of the DataFrame. 

    Parameters :
    --------------------------
    
        Kind of translator : GoogleAPI or LLM etc.
    
    """

    def __init__(self, TranslatorKind='LLM', llm_model="llama3.2:3b", batch_mode=True, chunk_size=2000):

        self.TranslatorKind=TranslatorKind
        self.batch_mode = batch_mode
        self.chunk_size = chunk_size

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
    # --------------- Translation Dispatcher --------------
    # -----------------------------------------------------

    def TxtTranslate(self, df, columns):

        df = df.copy()
        print(f"...Starting translation for {len(df)} rows with chunk size={self.chunk_size}...")

        # Split into chunks for large DataFrames
        print('-'*50)
        chunks = np.array_split(df, max(1, len(df)//self.chunk_size))
        print('-'*50)

        for i, chunk_df in enumerate(chunks, start=1):
            print('-'*50)
            print(f"Processing chunk {i}/{len(chunks)} ({len(chunk_df)} rows)")
            print('-'*50)
            if self.TranslatorKind == 'LLM':
                chunk_dfOut = self.translateByLLMBatch(chunk_df, columns)
            elif self.TranslatorKind == 'Google':
                chunk_dfOut = self.translateByGoogle(chunk_df,columns)
            else:
                print("⚠️ Unsupported TranslatorKind.")
                continue

            # Merge translated colums back into original df using index

            for col in columns:
                transformed_col = f"{col}_LLM" if self.TranslatorKind == 'LLM' else f"{col}_GGL"
                if transformed_col in chunk_dfOut.columns:
                    df.loc[chunk_dfOut.index, transformed_col] = chunk_dfOut[transformed_col]
            
        
        suffix = '_LLM' if self.TranslatorKind == 'LLM' else '_GGL'
        print(f"✅ All chunks processed. Total translated rows: {df[columns[0] + suffix].notnull().sum()}")

        return df
            
    # -----------------------------------------------------
    # --------------- Batch grouping utility --------------
    # -----------------------------------------------------

    def define_batches(self, chunk_df, columns):

        """
        Define batching groups based on word count in a text column.
        Returns a list of dictionaries defining masks and batch sizes.
        """
        chunk_df["nWords"] = chunk_df[columns].apply(lambda x: len(str(x).split()))

        groups = [
            {"name": "short", "mask": chunk_df["nWords"] < 200, "batch_size": 10},
            {"name": "medium", "mask": (chunk_df["nWords"] >= 200) & (chunk_df["nWords"] < 500), "batch_size": 3},
            {"name": "long", "mask": (chunk_df["nWords"] >= 500) & (chunk_df["nWords"] < 2000), "batch_size": 2},
            {"name": "very_long", "mask": chunk_df["nWords"] >= 2000, "batch_size": 1},
        ]
        chunk_df.drop(["nWords"], axis=1, inplace=True)
        return groups

    ####################################################
    ###################### LLM  ########################
    ####################################################
    
    # -----------------------------------------------------
    # ----------- LLM Transformation in Batch -------------
    # -----------------------------------------------------

    def ollamaBatchTransformation(self, inText, prompt, batch=False):
        """Handles both single and batch mode LLM calls."""
        if batch:
            batch_text = ";\n".join([f"{i+1}.. {t}" for i, t in enumerate(inText)])
            full_prompt = (
                f"{prompt}\n"
                f"Here are the texts:\n{batch_text}\n")
            response = ollama.chat(model=self.model, messages=[{"role": "user", "content": full_prompt}])
            content = response["message"]["content"].strip()
            # Comment - Uncomment following code lines and check the outcome
            matches = re.findall(r'(\d+)\.+\s(.*?)(?=\n\d+\.+|\Z)', content, flags=re.S)
            #matches = re.findall(r'(\d+)[\.\)\-]+[\s]*(.*?)(?=\n\d+[\.\)\-]+|\Z)', content, flags=re.S)
            #matches_dict = {str(key): value.strip() for key, value in matches}
            matches_dict = {key: value for key, value in matches}

            # if dictionary is single text then
            if not matches_dict and len(inText) == 1:
                matches_dict = {"1": content.strip()}

            return matches_dict, content
        else:
            batch_text = str(inText)
            full_prompt = (
                f"{prompt}\n"
                f"Here are the texts:\n{batch_text}\n"
                f"Keep the line numbering 1-N unchaged")
            response = ollama.chat(model=self.model, messages=[{"role": "user", "content": full_prompt}])
            return response["message"]["content"].strip()


    def translateByLLMBatch(self, chunk_df, columns):
        """
        Dynamically translate DataFrame rows by grouping based on text length (word count).
        Each range of text length uses its own batch size.
        """
        #prompt = input("Enter your prompt:\n ")
        prompt = """For each line:
        - Translate to English, Keep the input text line numbering 1-N unchaged.
        - Max 200 words for each line.
        - Do not translate text in english, just summarized.
        - Only factual attributes (product type, features, notable characteristics)
        - No marketing fluff, no repetition. 
        - Do not output labels, headings, bullet points, or extra commentary.
        """
        chunk_df = chunk_df.copy()

        # --------- Non-batch mode --------------

        if not self.batch_mode:
            for col in columns:
                if col not in chunk_df.columns:
                    print(f"Column '{col}' not found in DataFrame. Skipping.")
                    continue
                chunk_df[col] = chunk_df[col].fillna('Empty').replace('', 'Empty')
                chunk_df[f"{col}_LLM"] = chunk_df.progress_apply(
                    lambda row: self.ollamaBatchTransformation(row[col], prompt, batch=False),
                    axis=1)
            return chunk_df
        
        # --------- Batch mode -----------------

        #all_translated_parts = []
        for col in columns:
            if col not in chunk_df.columns:
                print(f"Warning: Column '{col}' not found. Skipping..")
                continue

            groups = self.define_batches(chunk_df,col)
            new_col = f"{col}_LLM"
            chunk_df[col] = chunk_df[col].fillna('Empty').replace('', 'Empty')
            # Step 3: Loop over groups
            for g in groups:
                sub_df = chunk_df[g["mask"]].copy()
                if sub_df.empty:
                    continue

                batch_size = g["batch_size"]
                n_batches = (len(sub_df) + batch_size - 1) // batch_size
                print(f"\nProcessing group '{g['name']}' ({len(sub_df)} rows) with batch size {batch_size} ({n_batches} batches)")

                # Step 4: Batch processing
                for batch_number, batch_df in tqdm(sub_df.groupby(np.arange(len(sub_df)) // batch_size)):
                    try:
                        #print(f"The batch number, {batch_number+1}, is being translated")
                        batch_texts = batch_df[col].astype(str).tolist()
                        #time.sleep(1)
                        translated, _ = self.ollamaBatchTransformation(batch_texts, prompt, batch=True)

                        # Retry if incomplete
                        max_retries = 2
                        retry_count = 0
                        while len(translated) < len(batch_df) and retry_count < max_retries:
                            retry_count += 1
                            print(f"⚠️ Batch {batch_number} : Translation incomplete")
                            print(f"{len(translated)}/{len(batch_df)}). Retrying {retry_count}/{max_retries}...")
                            #time.sleep(2)
                            translated, _ = self.ollamaBatchTransformation(batch_texts, prompt, batch=True)

                            if len(translated) == len(batch_df):
                                print(f"✅ Batch {batch_number} translated successfully on retry {retry_count}.")

                        if len(translated) < len(batch_df):
                            print(f" ❌ Batch {batch_number}: Translation still incomplete after {max_retries} retries. ")

                        sub_df.loc[batch_df.index, new_col] = [
                            translated.get(str(i+1), None) for i in range(len(batch_df))
                        ]

                    except Exception as e:
                        print(f"Error in batch {batch_number}: {e}")

                chunk_df.loc[sub_df.index, new_col] = sub_df[new_col]
                #all_translated_parts.append(sub_df)

        return chunk_df #pd.concat(all_translated_parts).sort_index()

    
    ####################################################
    ################ Google Translate  #################
    ####################################################     
    
    def _google_request(self, inText):
        if pd.isnull(inText) or str(inText).strip() == "":
            return inText
        try:
            return GoogleTranslator(source='auto', target='en').translate(inText)
        except Exception as e:
            with open("untranslated.txt", "a", encoding="utf-8") as f:
                f.write(f"{e}\n")
            self.error_count += 1
            return inText            
    
    def translateByGoogle(self, chunk_df, columns):

        chunk_df = chunk_df.copy()
        self.error_count = 0

        # --------- Non-batch mode --------------

        if not self.batch_mode:
            for col in columns:
                if col not in chunk_df.columns:
                    print(f"Column '{col}' not found in DataFrame. Skipping.")
                    continue
                out_col = f"{col}_GGL"
                chunk_df[out_col] = chunk_df[col].progress_apply(self._google_request)
            print(f"\n Total sentence more than 5000 character, {self.error_count}")
            return chunk_df

        # --------- Batch Model ----------------                    

        
        all_translated_parts = []  # to store translated DataFrames

        for col in columns:
            if col not in chunk_df.columns:
                print(f"Warning : Column '{col}' not found. Skipping..")
                continue

            groups = self.define_batches(chunk_df, col)
            new_col = f"{col}_GGL"
    
            # Step 3: Loop over groups
            for g in groups:
                sub_df = chunk_df[g["mask"]].copy()
                if sub_df.empty:
                    continue

                batch_size = g["batch_size"]
                n_batches = (len(sub_df)+batch_size-1)//batch_size
                print(f"\nProcessing group '{g['name']}' ({len(sub_df)} rows) with batch size {batch_size} ({n_batches} batches)")
                
                # Step 4: Batch Processing
                
                for batch_number, batch_df in tqdm(sub_df.groupby(np.arange(len(sub_df))//batch_size)):
                    #print(f"The batch number, {batch_number+1}, is being translated")
                    #print("The batchsize:",len(batch_df))
                    batch_df[new_col] = batch_df[col].apply(self._google_request)
                    sub_df.loc[batch_df.index, new_col] = batch_df[new_col]


                all_translated_parts.append(sub_df)
        print(f"\n Total sentence more than 5000 character, {self.error_count}")

        # step 5 : Merge all parts back into one frame 
        chunk_df = pd.concat(all_translated_parts).sort_index()

        return chunk_df