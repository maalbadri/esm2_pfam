# Protein Sequence Classification Project

This is an experimental model fine-tuned from the 
[esm2_t6_8M_UR50D](https://huggingface.co/facebook/esm2_t6_8M_UR50D) model 
for multi-label classification. In particular, the model is fine-tuned on the Pfam database available 
[here](https://www.ebi.ac.uk/interpro/), the general purpose of the Pfam database
is to provide a complete and accurate classification of protein families and domains.

In this project, fine-tuning refers to the process of training a pre-trained transformer model on a new dataset related to protein sequence classification. The pre-trained model, such as the ESM-2 transformer, has previously learned rich representations of biological sequences from a vast and diverse dataset. However, to make it useful for a specific classification task, we fine-tune it on a smaller dataset containing labeled protein sequences specific to our classification problem in Pfam.

## Trained models

Two trained models are available for download, one with classes capped at a minimum of 500 samples ("small" and has 290 classes) and one with classes capped at a minimum of 500 samples ("big" and has 1158 classes). Outputs for each model are in the `SMALL_MODEL` and `BIG_MODEL` directories, respectively.

These are automatically pulled from 🤗 during the `Evaluate.sh` command:

 - For the small model, see the HuggingFace model [here](https://huggingface.co/maalbadri/esm2_t6_8M_UR50D-finetuned-localization-small).
 - For the big model, see the HuggingFace model [here](https://huggingface.co/maalbadri/esm2_t6_8M_UR50D-finetuned-localization-big).

The model architecture of the original ESM-2 transformer is illustrated as a PyTorch model in `architecture.txt`.

The performance on the test set for each model is as follows:

### Small Model (7.93M params)
|            | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| Accuracy   |           |        | 0.9948   | 25513   |
| Macro Avg  | 0.9949    | 0.9940 | 0.9944   | 25513   |
| Weighted Avg| 0.9949   | 0.9948 | 0.9948   | 25513   |

### Big Model (8.21M params)
|            | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| Accuracy   |           |        | 0.8360   | 58490   |
| Macro Avg  | 0.7924    | 0.7476 | 0.7327   | 58490   |
| Weighted Avg| 0.8276   | 0.8360 | 0.8024   | 58490   |

## Data Processing

This project includes scripts for processing training data, fine-tuning a transformer model, and evaluating the model on a test set.

1. **Process Training Data:**
   - Script: `Preprocess.py`
   - Bash Script: `Preprocess.sh`
   - Description: Reads raw data, removes classes with fewer than a specified number of samples, and preprocesses sequences for training.
   - Usage:
     ```bash
     ./Preprocess.sh
     ```
## Training

2. **Train Model:**
   - Script: `finetune_esm2_to_pfam_instadeep.py`
   - Bash Script: `Train.sh`
   - Description: Fine-tunes a transformer model (facebook/esm2_t6_8M_UR50D) on the preprocessed training data.
   - Usage:
     ```bash
     ./Train.sh
     ```

## Evaluation

3. **Evaluate Model:**
   - Script: `Evaluate.py`
   - Bash Script: `Evaluate.sh`
   - Description: Downloads the fine-tuned model pushed during step 2, and evaluates it on a test set, generating a dataframe of predictions, a classification report and a confusion matrix to the cwd.
   - Usage:
     ```bash
     ./Evaluate.sh
     ```

## Dependencies

An environment file is provided for conda users. To create a conda environment with the required dependencies, run the following command:

```bash
conda env create -f instadeep_env.yml
```

## Author

- [Dr. Mohamed Ali al-Badri](https://github.com/maalbadri) - Research Fellow at [UCL](https://www.ucl.ac.uk/).

## Citations
This work is based on the following work:

```bibtex
@article{lin2023evolutionary,
  title={Evolutionary-scale prediction of atomic-level protein structure with a language model},
  author={Lin, Zeming and Akin, Halil and Rao, Roshan and Hie, Brian and Zhu, Zhongkai and Lu, Wenting and Smetanin, Nikita and Verkuil, Robert and Kabeli, Ori and Shmueli, Yaniv and others},
  journal={Science},
  volume={379},
  number={6637},
  pages={1123--1130},
  year={2023},
  publisher={American Association for the Advancement of Science}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.