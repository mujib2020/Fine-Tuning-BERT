# BERT Fine-Tuning with Keras for Binary Text Classification

This repository provides a comprehensive guide on how to fine-tune the BERT base model for binary text classification tasks using the Keras library in TensorFlow. 

The tutorial covers all the steps needed to adapt the powerful BERT model to an specific text data, enabling improved classification performance due to BERT's deep understanding of language nuances.

## Tutorial Overview

- Introduction to BERT and its advantages for NLP tasks.
- Setting up the environment and dependencies.
- Preparing the text dataset for fine-tuning.
- Loading and configuring the pre-trained BERT model with Keras.
- Training the model with a dataset from Hugging face: https://huggingface.co/datasets/glue.
- Evaluating model performance on a test set. (Note: the test set doesn't have the true label)
- Analyze/Visualize the predictions on un seen test dataset. 

### Introduction to BERT and its advantages for NLP tasks:
BERT (Bidirectional Encoder Representations from Transformers) is a powerful language representation model that has revolutionized the way machines understand human language.

### Setting up the environment and dependencies.

pip install datasets

pip install transformers

from datasets import load_dataset

from transformers import AutoTokenizer

import tensorflow

from transformers import TFAutoModelForSequenceClassification

from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt

**For rest of the section**, please check either **Fine_Tune_BERT_1_.ipynb** or **Fine_Tune_BERT_1_.py**
