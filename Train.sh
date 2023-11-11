#!/bin/bash

# Set your desired values for the arguments
MODEL_CHECKPOINT="facebook/esm2_t6_8M_UR50D"
TRAIN_DATA_PATH="train_data.csv"
BATCH_SIZE=4
LEARNING_RATE=2e-5
NUM_EPOCHS=1
MODEL_OUTPUT_NAME='big' # 'small'
# Make sure you generate a token with WRITE access if
# you want to upload your model checkpoint to the Hugging Face model hub
HUGGINGFACE_TOKEN=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
PUSH_TO_HUB=true

python finetune_esm2_to_pfam_instadeep.py \
  --model_checkpoint $MODEL_CHECKPOINT \
  --train_data_path $TRAIN_DATA_PATH \
  --batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --num_epochs $NUM_EPOCHS \
  --push_to_hub \
  --model_output_name $MODEL_OUTPUT_NAME \
  --huggingface-token $HUGGINGFACE_TOKEN 
