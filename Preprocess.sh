#!/bin/bash

# Set your desired values for the arguments, 
# note that MAX_SEQ_LEN ≤ 1024.

#MIN_CLASS_SIZE=500
MIN_CLASS_SIZE=200
MAX_SEQ_LEN=1024
PATH_TRAIN="train_data.csv"
PATH_TEST="test_data.csv"

# Run the Python script with the specified arguments
python Preprocess.py --min_class_size $MIN_CLASS_SIZE --max_seq_len $MAX_SEQ_LEN --path_train $PATH_TRAIN --path_test $PATH_TEST
