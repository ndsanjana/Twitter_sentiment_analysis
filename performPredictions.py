# Load model directly
import nltk
import re
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def preprocess_text(text):
    # Remove URLs, mentions, hashtags, and special characters
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords and lemmatize the tokens
    tokens = [
        lemmatizer.lemmatize(word.lower())
        for word in tokens
        if word.lower() not in stop_words
    ]
    return " ".join(tokens)


def prepareData(test):
    # Apply preprocessing to the text data
    test["full_text"] = test["full_text"].apply(preprocess_text)
    testLabels = np.array(test["label"])
    test["Right"] = test["label"] == 0
    test["Left"] = test["label"] == 1
    test["Neutral"] = test["label"] == 2
    test.drop(columns=["label"], inplace=True)

    labels = ["Right", "Left", "Neutral"]
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}
    return test, id2label, label2id, testLabels


# Inference loop
def performInference(model, test, testLabels):
    all_preds = []
    for i in range(len(test)):
        batch = tokenizer(
            test.iloc[i]["full_text"],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits

        # Apply sigmoid and threshold at 0.5
        probs = torch.sigmoid(logits.squeeze().cpu())
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs >= 0.5)] = 1
        all_preds.append(predictions)
        # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_preds = np.argmax(all_preds, axis=1)

    # Calculate F1 score (macro-average for multi-label)
    f1 = f1_score(testLabels, all_preds, average="macro")
    # Print confusion matrix
    cm = confusion_matrix(testLabels, all_preds)

    return f1, cm


def performHoldbackPredictions(holdback, model):
    predictions_list = []
    for i in range(len(holdback)):
        batch = tokenizer(
            holdback.iloc[i]["full_text"],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits

        # Apply sigmoid and threshold at 0.5
        probs = torch.sigmoid(logits.squeeze().cpu())
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs >= 0.5)] = 1
        prediction = np.argmax(predictions, axis=0)
        predictions_list.append({"id": holdback.iloc[i]["id"], "label": prediction})
    return pd.DataFrame(predictions_list)


tokenizer = AutoTokenizer.from_pretrained(
    "rahulk98/bert-finetuned-twitter_sentiment_analysis"
)
model = AutoModelForSequenceClassification.from_pretrained(
    "rahulk98/bert-finetuned-twitter_sentiment_analysis"
)

# Download necessary NLTK data
nltk.download("stopwords")
nltk.download("punkt_tab")
nltk.download("wordnet")

# Initialize stopwords, tokenizer, and lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


train = pd.read_csv("train.csv", sep=",")


test = train.sample(frac=0.1)
train = train.drop(test.index)

test, id2label, label2id, testLabels = prepareData(test)
max_length = test["full_text"].str.len().max()
print(f"Maximum text length: {max_length}")

f1_score, cm = performInference(model, test, testLabels)
print("Results on test set with BERT Fine Tuned model: ")
print(f"F1 score: {f1_score}")
print(f"Confusion matrix:\n{cm}")

holdback = pd.read_csv("holdback_noLabel.csv", sep=",")

holdback["full_text"] = holdback["full_text"].apply(preprocess_text)

results_df = performHoldbackPredictions(holdback, model)
# Save predictions to CSV
results_df.to_csv("predictions.csv", index=False)
