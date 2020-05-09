This page contains scripts for the final project of the EMNLP class - Spring 2020 by Shabnam Behzad, Sajad Sotudeh, Shira Wein. 

The [data](data/) directory contains a sample of the data in the required format. In the [models](models/) directory, you can find the scripts for each model separately. In what follows, we will explain how to run each model.


## LM

This model uses the Hugging Face pre-trained model bert-base-cased. No training data is needed for this model. You need to use the jsonl format for this script. The only required arguments are the path to test data (jsonl) and the path to the directory in which you would like to save the results.
You can simply run the code with the following command:

```
python bert-lm.py -t path_to_jsonl_test_file -o path_to_results_directory
```

## SMT

This model requires installation of the Natural Language Toolkit (NLTK(. Installation instructions can be found here: https://www.nltk.org/install.html

You will also need to download train.json and dev.json (found in models/smt), since those are the files used in this experimentation. Once you have the files downloaded and NLTK installed, you can simply run main.py. No arguments are required. For additional experiments other json files could be used.

## Sequence-to-sequence

# Requirements

- Python 3
- PyTorch
- [tqdm](https://github.com/tqdm/tqdm)
- [errant](https://github.com/chrisjbryant/errant)
- unzip, wget (for downloading only)
- nltk

## Data
Download the clean data (wi+locness dataset) under `dataset/` directory. You will find json files where each line shows a single instance with keys:
- `src: list()`: The incorrect form of the sentence.
- `tgt: list()`: The correct form of the sentence.
- `split: str`: The split that the sentence has been sampled from.

## Training

### Preparation

The model is initialized with Common Crawl GloVe word vectors. First, you have to download these vectors by running:
```
sh download.sh
```

Preparing vocab, considering that `$GEC` is a directory containing the GEC splits:
```
python prepare_vocab.py dataset/$GEC dataset/vocab --glove_dir dataset/glove
```

This will write vocabulary and word vectors as a numpy matrix into the dir `dataset/vocab`.

### Run training

To start training on  data, run:
```
python train.py --id $ID --data_dir dataset/$GEC
```

This will train a sequence-to-sequence model with copy mechanism and save everything into the `saved_models/$ID` directory. `$ID` is the model identifier that you wish to save.

*Note:* You can start training by `sh train.sh` alternatively.

The valdation step will be done after completion of each epoch. The default epochs are set to be 30 in the model. So, we will have 30 evaluation steps.


Please refer to the readme placed in the [model directory](models/seq-to-seq) for the detailed instructions.
