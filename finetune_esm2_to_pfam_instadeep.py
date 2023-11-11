import argparse

import pickle
import pandas as pd
import numpy as np
import torch

from transformers import AutoTokenizer
from datasets import Dataset

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from evaluate import load


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# clear cuda memory
import torch
import gc
gc.collect()
torch.cuda.empty_cache()

# seed everything for reproducibility:
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', type=str, required=True, help='Hugging Face model checkpoint')
    parser.add_argument('--train_data_path', type=str, required=True, help='Path to the training data CSV file')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for the optimiser')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--model_output_name', type=str, default="small", help='Suffix of the model output name')
    parser.add_argument('--huggingface-token', required=False, help='Hugging Face token, if none is provided, the script will prompt you to log in.')
    parser.add_argument('--push_to_hub', action='store_true', help='Push the model checkpoint to the Hugging Face hub')
    opts = parser.parse_args()
    return opts

def main():
    args = get_args()
        
    # huggingface_hub login:
    # You will need an account to access the model checkpoint,
    # and to subsequently push the model checkpoint to your own huggingface_hub account.
    from huggingface_hub import login
    if args.huggingface_token:
        login(token=args.huggingface_token)
    else:
        login()

    # the model to finetune is an ESM2 transformer model, which is a pretrained model on UniRef50 sequences:
    # See https://huggingface.co/facebook/esm2_t6_8M_UR50D for more details.
    model_checkpoint = "facebook/esm2_t6_8M_UR50D"

    # the training data has already been pre-processed in a separate script
    # load it here:
    df = pd.read_csv("train_data.csv")

    # the dataframe has two columns: "sequence" and "labels", the labels are not encoded yet
    # encode the labels (strings) into integers:
    label_encoder = LabelEncoder()
    df['labels'] = label_encoder.fit_transform(df['labels'])
    # save the label encoder for later use in inference.py

    # Loop over the classes and make a dictionary with the class names and their corresponding integer values:
    label_dict = {}
    for index, label in enumerate(label_encoder.classes_):
        label_dict[index] = label
        
    # the sklearn labelencoder object can be used to transform 
    # the integer encoded labels back into the original string labels: 
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    # get all sequences and put them in a list:
    sequences = df["sequence"].tolist()
    labels = df["labels"].tolist()

    assert len(sequences) == len(labels), "The number of sequences and labels are not equal."
    train_sequences, test_sequences, train_labels, test_labels = train_test_split(sequences, labels, test_size=0.25, shuffle=True)

    # tokenize the sequences:
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    train_tokenized = tokenizer(train_sequences)
    test_tokenized = tokenizer(test_sequences)

    # create the dataset:
    train_dataset = Dataset.from_dict(train_tokenized)
    test_dataset = Dataset.from_dict(test_tokenized)

    # add the labels to the dataset:
    train_dataset = train_dataset.add_column("labels", train_labels)
    test_dataset = test_dataset.add_column("labels", test_labels)

    num_labels = max(train_labels + test_labels) + 1 
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

    model_name = model_checkpoint.split("/")[-1]
    # batch_size = 4

    # define the training arguments to pass to the Trainer class:
    args = TrainingArguments(
        f"{model_name}-finetuned-localization-{args.model_output_name}",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=args.push_to_hub,
    )

    metric = load("accuracy")

    # make a function to compute the metrics:
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    # initialise the trainer:
    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # initiate the training:
    trainer.train()

if __name__ == "__main__":
    main()
