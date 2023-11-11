import argparse
import pickle
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path_directory', type=str, required=True, help='Directory path to the fine-tuned model')
    parser.add_argument('--test_data_path', type=str, required=True, help='Path to the test data CSV file')
    parser.add_argument('--label_encoder_path', type=str, required=True, help='Path to the LabelEncoder pickle file')
    parser.add_argument('--output_csv', type=str, default='df_test_preds.csv', help='Path to save the output CSV file')
    parser.add_argument('--output_classification_report', type=str, default='classification_report.txt', help='Path to save the classification report text file')
    parser.add_argument('--output_confusion_matrix', type=str, default='confusion_matrix.png', help='Path to save the confusion matrix image')
    opts = parser.parse_args()
    return opts

def predict_category(sequence, tokenizer, model):
    inputs = tokenizer(sequence, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

def main():
    args = get_args()

    # Initialise the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path_directory)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path_directory)

    # load the test set:
    df_pfam = pd.read_csv(args.test_data_path)

    # load the labelencoder dictionary:
    with open(args.label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    predicted_category = []
    true_category = []

    for index, row in tqdm(df_pfam.iterrows(), total=df_pfam.shape[0]):
        sequence = row["sequence"]
        label = row["labels"]
        predicted_class = predict_category(sequence, tokenizer, model)
        predicted_category.append(predicted_class)
        true_category.append(label_encoder.transform([label])[0])

    df_pfam["predicted_category"] = predicted_category
    df_pfam["true_category"] = true_category

    # save the dataframe to a csv file:
    df_pfam.to_csv(args.output_csv, index=False)

    # get classification report from sklearn
    true_category = df_pfam["true_category"].tolist()
    predicted_category = df_pfam["predicted_category"].tolist()

    # save the classification report to a txt file:
    with open(args.output_classification_report, "w") as f:
        f.write(classification_report(true_category, predicted_category, digits=4, zero_division=0))

    # draw a confusion matrix:
    cm = confusion_matrix(true_category, predicted_category)
    plt.figure(figsize=(30, 30))
    sns.heatmap(cm, annot=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # remove x/y tick labels
    plt.xticks([])
    plt.yticks([])

    # save the confusion matrix, this looks silly for huge # of classes 
    plt.savefig(args.output_confusion_matrix, dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    main()
