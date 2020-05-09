Grammatical Error Correction
==========

This repo contains the PyTorch code and pretrained model for the ENLP project, Spring 2020, Georgetown University.


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
sh download.sh; ./download.sh
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