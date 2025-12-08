import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from transformers import BertTokenizer, BertModel
from transformers import AutoModelForSequenceClassification
from torch.utils.data import WeightedRandomSampler
from collections import Counter
from PIL import Image
from tqdm import tqdm
from datasets import load_from_disk
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms.functional import to_pil_image
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

import os
import zipfile
import random

class DemoDataset(Dataset):
    def __init__(self, bert_path, num_samples=None):
        """
        samples:
                 {"text": "...", "image": "...", "label": ...}
        """
        samples = load_from_disk("./data/preprocessed/fusion")
        test_set = samples["test"]

        if num_samples:
            indices = random.sample(range(len(test_set)), num_samples)
            self.samples = test_set.select(indices)
        else:
            self.samples = test_set

        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.transform =    transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),  # MUST come first
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # TEXT
        encoded = self.tokenizer(
            s["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].squeeze()
        attention_mask = encoded["attention_mask"].squeeze()

        # IMAGE
        img = Image.open(s["image"]).convert("RGB")
        img = self.transform(img)

        label = torch.tensor(s["label"], dtype=torch.long)

        return input_ids, attention_mask, img, label
    

class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes, effnet_path, dropout=0.5):
        super(EfficientNetClassifier, self).__init__()
        
        # Load pretrained EfficientNet-B0
        self.base_model = models.efficientnet_b0(weights=None)

        self.num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.num_features, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)
        )

        state_dict = torch.load(effnet_path, map_location="cpu")
        new_state = {}
        for k, v in state_dict.items():
            new_k = k.replace("base_model.", "")
            new_state[new_k] = v

        self.base_model.load_state_dict(new_state, strict=True)
        
    def forward(self, x):
        return self.base_model(x)
    
    def extract_features(self, x):
        x = self.base_model.features(x)
        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def freeze_base(self):
        """Freeze all base model layers except classifier"""
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def unfreeze_top_layers(self, num_layers=50):
        """Unfreeze top N layers for fine-tuning"""
        layers = list(self.base_model.features.children())
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

# Fusion Model
class LateFusionModel(nn.Module):
    def __init__(self, bert_path, effnet_path, num_classes):
        super(LateFusionModel, self).__init__()

        self.bert = AutoModelForSequenceClassification.from_pretrained(
            bert_path, num_labels=27
        )
        self.bert_hidden = self.bert.config.hidden_size

        for p in self.bert.parameters():
            p.requires_grad = False

        self.effnet = EfficientNetClassifier(
            num_classes=27, effnet_path=effnet_path, dropout=0.5
        )
        self.effnet.freeze_base()
        self.effnet_hidden = self.effnet.num_features

        self.classifier = nn.Sequential(
            nn.Linear(self.bert_hidden + self.effnet_hidden, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_ids, attention_mask, image):
        bert_output = self.bert.base_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        text_feat = bert_output.pooler_output

        img_feat = self.effnet.extract_features(image)

        fused = torch.cat([text_feat, img_feat], dim=1)
        out = self.classifier(fused)
        return out
    
class Inference:
    def __init__(self, fusion_path="./models/fusion-model/fusion_best_V2.pth", bert_path="./models/bert-model", effnet_path="./models/effnet-model/best_model_final.pth", num_classes=27, batch_size=6):
        
        self.bert_path = bert_path
        self.effnet_path = effnet_path
        self.fusion_path = fusion_path
        self.batch_size = batch_size
        
        self.ds = DemoDataset(bert_path=self.bert_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_loader = DataLoader(self.ds, batch_size=self.batch_size, shuffle=False)
        self.fusion_model = LateFusionModel(bert_path=self.bert_path, effnet_path=self.effnet_path, num_classes=27).to(self.device)
        self.fusion_model.load_state_dict(torch.load(self.fusion_path, map_location=torch.device('cpu')))
        self.fusion_model.eval()

    def select_random_samples(self):
        self.ds = DemoDataset(bert_path=self.bert_path, num_samples=self.batch_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_loader = DataLoader(self.ds, batch_size=self.batch_size, shuffle=False)

    def load_id2label(self, file_path="./inference/id2label"):

        with open(file_path, "r", encoding="utf-8") as f:
            self.id2_label = json.load(f)
    
    def id2_label(self, id):
        return self.id2_label[id]

    def reload(self):

        self.ds = DemoDataset(bert_path=self.bert_path)
        self.data_loader = DataLoader(self.ds, batch_size=self.batch_size, shuffle=False)

    def unnormalize(self, img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        img = img_tensor.clone()
        for t, m, s in zip(img, mean, std):
            t.mul_(s).add_(m)
        return img
    
    def display_cm(self, labels, predicted_classes):

        class_report = classification_report(labels, predicted_classes)
        print("Classification Report:\n", class_report)

        print("\nConfusion Matrix:")
        cm = confusion_matrix(labels, predicted_classes)

        self.load_id2label()
        ticks = []
        for i in range(27):
            ticks.append(self.id2_label[str(i)])

        plt.figure(figsize=(20, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=ticks, yticklabels=ticks)
        plt.title("Confusion Matrix")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.show()
        
    def test_metrics(self):
        self.select_random_samples()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for input_ids, attention_mask, images, labels in self.data_loader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.fusion_model(input_ids=input_ids, attention_mask=attention_mask, image=images)

                probs = torch.softmax(logits, dim=1).cpu().numpy() 
                preds = probs.argmax(axis=1)

                all_labels.extend(labels)
                all_preds.extend(preds)

            self.display_cm(all_labels, all_preds)             

    
    def predict(self, reload=False, visualize=False):

        self.select_random_samples()

        if reload:
            self.reload()

        self.load_id2label()

        with torch.no_grad():
            for input_ids, attention_mask, images, labels in self.data_loader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.fusion_model(input_ids=input_ids, attention_mask=attention_mask, image=images)

                probs = torch.softmax(logits, dim=1).cpu().numpy() 
                preds = probs.argmax(axis=1)                    

                if not visualize:
                    # Print predictions + true labels for each sample
                    for true_label, pred_label in zip(labels.cpu().numpy(), preds):
                        print(f"True: {self.id2_label[true_label.astype(str)]}, Predicted: {self.id2_label[pred_label.astype(str)]}")

                else: 
                    input_ids_cpu = input_ids.cpu()
                    images_cpu = images.cpu()
                    labels_cpu = labels.cpu().numpy()

                    for i in range(len(labels_cpu)):
                        text = self.ds.tokenizer.decode(input_ids_cpu[i], skip_special_tokens=True)
                        true_label = labels_cpu[i]
                        pred_label = preds[i]
                        img = self.unnormalize(images_cpu[i])
                        img = to_pil_image(img)

                        # --- Create dark-themed figure ---
                        fig, ax = plt.subplots(1, 2, figsize=(8, 6))
                        fig.patch.set_facecolor("#1a1a1a")   # Dark outer background

                        # -------------------------------------------------
                        # LEFT PANEL — IMAGE
                        # -------------------------------------------------
                        ax[0].imshow(img)
                        ax[0].set_title(
                            "Input Image",
                            fontsize=18,
                            color="white",
                            pad=15
                        )
                        ax[0].axis("off")

                        # Add elegant white frame
                        rect = patches.Rectangle(
                            (0, 0), 1, 1,
                            linewidth=2.0,
                            edgecolor="white",
                            facecolor="none",
                            transform=ax[0].transAxes
                        )
                        ax[0].add_patch(rect)

                        # -------------------------------------------------
                        # RIGHT PANEL — TEXT SECTION
                        # -------------------------------------------------
                        ax[1].set_facecolor("#1a1a1a")  # Panel background dark
                        ax[1].axis("off")

                        # Prediction summary
                        info_title = (
                            f"True:         →   {self.id2_label[true_label.astype(str)]}\n"
                            f"Predicted:    →   {self.id2_label[pred_label.astype(str)]}"
                        )

                        if true_label == pred_label:
                            ax[1].text(
                                0.01, 0.95,
                                info_title,
                                fontsize=16,
                                fontweight="bold",
                                va="top",
                                ha="left",
                                color="green"
                            )

                        else:  
                            ax[1].text(
                                0.01, 0.95,
                                info_title,
                                fontsize=16,
                                fontweight="bold",
                                va="top",
                                ha="left",
                                color="red"
                            )

                        # Text label
                        ax[1].text(
                            0.01, 0.85,
                            "Input Text:",
                            fontsize=14,
                            fontweight="bold",
                            va="top",
                            ha="left",
                            color="white"
                        )

                        # Text box with light background
                        ax[1].text(
                            0.01, 0.80,
                            text,
                            fontsize=12,
                            family="monospace",
                            va="top",
                            ha="left",
                            wrap=True,
                            color="black",
                            bbox=dict(
                                facecolor="#e6e6e6",
                                edgecolor="#555555",
                                boxstyle="round,pad=0.6"
                            )
                        )

                        plt.tight_layout()
                        plt.show()
