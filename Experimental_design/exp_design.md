## Experimental Design: ESM2 vs. ProteinBERT for Protein Sequence Classification

### Introduction:

Large language models, trained on protein sequences, have demonstrated remarkable capabilities in capturing intricate biological features. Two prominent models for protein language modelling, ESM2 (Evolutionary Scale Modeling) and ProteinBERT, have gained attention for their ability to learn complex representations from biological data. In this set of experiments, the aim is to compare the performance of these models in predicting protein sequences for multiple classes for the Pfam dataset. The classes are determined by the threshold of the minimum number of samples per class. The ESM2 model is fine-tuned on the Pfam dataset, and the ProteinBERT model is trained on the same dataset. The models are then evaluated on a separate test dataset to assess their performance. The results of this experiment will provide insights into the strengths and weaknesses of each model in the context of protein sequence classification.

### Objectives:

1. **Evaluate Model Accuracy:**
   - Assess the accuracy of both ESM and ProteinBERT in predicting protein sequences for a given number of classes.

2. **Performance on Structurally Diverse Sequences:**
   - Examine how well each model performs on sequences with varying levels of structural complexity. Introduction of proteins from a variety of biological domains (membrane proteins, metallo-proteins, proteins with allostery, etc) will help evaluate the models' performance on structurally diverse sequences.

3. **Computational Efficiency:**
   - Compare the computational efficiency of the models, considering both training and inference times.

### Experimental Setup:

#### Data:

- Continue to utilise the Pfam dataset for training and evaluation.

#### Model Configuration:

1. **ESM Transformer:**
   - Fine-tune a pre-trained ESM model with a varying number of parameters (where possible; considering the high computational expense of fine-tuning or running inference on the multi-billion parameter models).
   - Apply the same pre-processing steps as in the current project for a fair comparison.

2. **ProteinBERT:**
   - Use the ProteinBERT model but adjust the classifier head to predict multiple classes in accordance with the number of classes in this project.
    - Apply the same pre-processing steps as in the current project for a fair comparison.

#### Experimental Procedures:

1. **Training:**
   - Train both models on the same training dataset, keeping the number of epochs consistent, else incorporating early stopping to prevent overfitting.
   - Monitor and record training performance metrics using Weights & Biases and/or Tensorboard, as in this project.
   - For rigorous fine-tuning and comparison experiment, I would run parameter sweeps to find the optimal hyperparameters for each model. This can be done using the [Weights & Biases Sweeps](https://docs.wandb.ai/guides/sweeps) feature.

2. **Evaluation:**
   - Assess the accuracy of both models on the same test dataset.
   - Evaluate the models' performance on sequences with varying levels of structural complexity to determine if one model excels in predicting protein structures for sequences with high perplexity.

3. **Computational Efficiency:**
   - Measure the training and inference times for both ESM2 and ProteinBERT models.
   - Compare the efficiency of the models in terms of computational resources.

### Expected Outcomes:

1. **Prediction Accuracy:**
   - I expect that the ESM transformer, with its evolutionary scale modeling, might outperform ProteinBERT in capturing complex biological patterns on an atomic scale and thus predict protein sequences with higher accuracy.

2. **Computational Efficiency:**
   - Investigate whether one model demonstrates superior computational efficiency, considering the scale and architecture differences. I expect that the ProteinBERT model will be more computationally efficient than the ESM2 transformer, considering the difference in the number of parameters.

### Conclusion:

This experimental design aims to provide insights into the strengths and weaknesses of the ESM2 transformer and ProteinBERT in the context of protein sequence classification.
