
import argparse
import numpy as np
import pandas as pd
import os

def read_data(name_sub_folder):
    full_data = []
    for f in os.listdir(os.path.join('random_split/random_split', name_sub_folder)):
        data = pd.read_csv(os.path.join('random_split/random_split', name_sub_folder, f))
        full_data.append(data)
    return pd.concat(full_data)  

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_class_size', type=int, default=200, help='minimum class size')
    parser.add_argument('--max_seq_len', type=int, default=1024, help='maximum sequence length, must be ≤ 1024')
    parser.add_argument('--path_train', type=str, default='train_data.csv', help='path to save train data')
    parser.add_argument('--path_test', type=str, default='test_data.csv', help='path to save test data')
    opts = parser.parse_args()
    return opts

def main():
    args = get_args()

    assert args.max_seq_len <= 1024, "The maximum sequence length must be ≤ 1024."

    df_train = read_data('train')

    # Find the counts of all the classes in the training set:
    counts_dict = df_train['family_accession'].value_counts().to_dict()

    # make a dataframe with the counts of all the classes in the training set:
    counts_df = pd.DataFrame.from_dict(counts_dict, orient='index')
    counts_df.columns = ['count']
    counts_df.index.name = 'class'

    MIN_CLASS_SIZE = args.min_class_size
    MAX_SEQ_LEN = args.max_seq_len - 2  # -2 for the special tokens

    # make a list of all the classes with fewer than MIN_CLASS_SIZE samples/class in the training set:
    classes_to_remove = counts_df[counts_df['count'] < MIN_CLASS_SIZE].index.tolist()

    print('-+'*20)
    print(f'Removing classes with fewer than {MIN_CLASS_SIZE} samples/class.')
    print(f'Classes removed: {len(classes_to_remove)}')
    print(f'Classes remaining {len(counts_df) - len(classes_to_remove)}')
    print('-+'*20)

    # remove all classes with fewer than 500 samples/class in the training set:
    df_train = df_train[~df_train['family_accession'].isin(classes_to_remove)]

    df_train = df_train[['sequence', 'family_accession']]
    df_train.columns = ['sequence', 'labels']

    # cap the sequence length at MAX_SEQ_LEN:
    cut_seq = lambda x: x[:MAX_SEQ_LEN] if len(x) > MAX_SEQ_LEN else x
    df_train['sequence'] = df_train['sequence'].apply(cut_seq)

    # save the df_train dataframe to a csv file:
    df_train.to_csv(args.path_train, index=False)

    # load the test set:
    df_test = read_data('test')
    df_test = df_test[['sequence', 'family_accession']]
    df_test.columns = ['sequence', 'labels']

    # remove the classes that we removed from the training set:
    df_test = df_test[~df_test['labels'].isin(classes_to_remove)]

    # cap the sequence length at 1024, as this is specified by the model:
    df_test['sequence'] = df_test['sequence'].apply(cut_seq)

    # reset the index and save the test set to a csv file:
    df_test.reset_index(drop=True)
    df_test.to_csv(args.path_test, index=False)

if __name__ == "__main__":
    main()
