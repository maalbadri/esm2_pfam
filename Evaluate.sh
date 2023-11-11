#!/bin/bash

# Set your desired values for the arguments
MODEL_NAME="big"
MODEL_PATH="maalbadri/esm2_t6_8M_UR50D-finetuned-localization-$MODEL_NAME"
TEST_DATA_PATH="test_data.csv"
LABEL_ENCODER_PATH="label_encoder.pkl"
OUTPUT_CSV="df_test_preds_$MODEL_NAME.csv"
OUTPUT_CLASSIFICATION_REPORT="classification_report_$MODEL_NAME.txt"
OUTPUT_CONFUSION_MATRIX="confusion_matrix_$MODEL_NAME.png"

# Run the Python script with the specified arguments
python Evaluate.py \
  --model_path_directory $MODEL_PATH \
  --test_data_path $TEST_DATA_PATH \
  --label_encoder_path $LABEL_ENCODER_PATH \
  --output_csv $OUTPUT_CSV \
  --output_classification_report $OUTPUT_CLASSIFICATION_REPORT \
  --output_confusion_matrix $OUTPUT_CONFUSION_MATRIX
