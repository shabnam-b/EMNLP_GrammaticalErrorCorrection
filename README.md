This page contains scripts for the final project of the EMNLP class - Spring 2020 by Shabnam Behzad, Sajad Sotudeh, Shira Wein. 

The [data](data/) directory contains a small sample of the data in the required format. In the [models](models/) directory, you can find the scripts for each model separately. In what follows, we will explain how to run each model.


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

Please refer to the readme placed in the [model directory](models/seq-to-seq) for the detailed instructions.
