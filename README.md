Rakuten Challenge MLOps
=======================

Project Context And Objective
-----------------------------

This Project is based on the Rakuten France dataset and aims to classify e-commerce products into 27 categories using both text descriptions and product images. 
Classifying products using both visual and textual descriptions - known as multi-modal product classification - is a core challenge in any e-commerce marketplace. This task supports critical applications such as personalized search, product recommendations, and query understanding.

The dataset from Rakuten France contains 84,916 products with multilingual text descriptions (French, German, English, etc.) with associated images.

Main Goal is to build a multi-modal deep learning model that effectively combines text and image data to accurately classify products into their respective categories.

See the following URL for more details: https://challengedata.ens.fr/challenges/35

Methodology
--------------------

We followed following methodology to examine the data, testing different models and set the final fusion model to enable product class prediction based on image and text:

Raw Data Input (Text + Image) -> Data Exploration -> Data Preprocessing -> Unimodal Models -> Multimodal Fusion Modal -> Product Class

### Data Exploration

We analyzed the raw text and image data to detect characteristics to considered for further processing like:
- severe class imbalance
- multilingual text
- missing text data
- high text variance
- HTML artifacts in text
- white margins in images

### Data Preprocessing

With the detected characteristics of the raw data, we preprocessed both text and image data to have a clean version for the model training.

Text
- removed HTML/XML entities
- removed special characters
- fixed malformed text
- removed outliers
- summarized text
- translated text

Image
- reduced white borders of images
- removed duplicates

### Unimodal Models

Further we examined different models for each text and image to find the best performing models.

- Text: Baseline (LSTM, Bi-LSTM, GRU) and Transfer-Learning (pretrained BERT model)
- Image: Baseline (Custom CNN) and Transfer-Learning (pretrained EfficientB0 model)

Due to the better performing transfer learning models, we kept those for the final fusion model.

### Multimodal Fusion Model

Here we combined both selected unimodal text and image model encoders to a multimodal late fusion architecture.

Results
--------------------

We achieved following F1-scores:
- text (BERT): 84 %
- image (EfficientNet-B0): 59 %
- multimodal late fusion: 83 %

Outlook
--------------------

Our work and findings can be used for further enhancement:
- improve model scores by investigating other model approaches and training strategies
- wrap preprocessing, model training and evaluation to a full data pipeline
- set a reproducible and scalable model architecture and deployment (MLOps)

Report And Streamlit
--------------------

For more detailed information about the project and its results, please read the report (./report/)
or run the Streamlit presentation:

```
# Move to streamlit folder
cd ./streamlit

# Run streamlit
streamlit run 1_Introduction.py
```

Repository Structure
--------------------

```
├── data                                            # Raw and preprocessed datasets
│   ├── preprocessed
│   │   ├── Example_Preprocessed_Images
│   │   ├── fusion
│   │   │   ├── test
│   │   │   ├── train
│   │   │   └── val
│   │   └── images
│   │       ├── image_test_preprocessed
│   │       └── image_train_preprocessed
│   ├── raw
│   │   ├── X_test.csv
│   │   ├── X_train.csv
│   │   ├── y_train.csv
│   │   └── images
│   │       ├── image_test
│   │       └── image_train
│   └── tmp
├── data_analysis                                   # Data exploration
├── data_model                                      # Saved final models for text, image and fusion
│   ├── bert-model
│   ├── effnet-model
│   └── fusion-model
├── data_preprocessing                              # Data preprocessing scripts
├── inference                                       # Inference
├── modeling                                        # Final modeling scripts (transfer learning) for text, image and fusion
│   ├── fusion
│   └── transfer_learning
│       ├── image
│       └── text
├── models_baseline                                 # Baseline modeling scripts
│   ├── baseline_text
│   └── baseline_image
├── report                                          # Project report
├── research                                        # Further research and trials
└── streamlit                                       # Streamlit project presentation
```

Setup Repository
--------------------

1. Clone remote repository to local
2. Setup virtual environment and activate (Python version here: 3.11)
3. Install all packages
4. Download Spacy model (`python -m spacy download en_core_web_sm`)
5. Download Natural Language toolkit (`python -m nltk.downloader all`)
6. Download raw data from https://challengedata.ens.fr/challenges/35 and save to ./data/raw
