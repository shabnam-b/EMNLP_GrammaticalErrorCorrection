This page contains scripts for the final project of the EMNLP class - Spring 2020


The data directory contains a small sample of the data in the required format. In the model directory, you can find the scripts for each model separately. In what follows, we will explain how to run each model.


## LM

This model uses the Hugging Face pre-trained model bert-base-cased. No training data is needed for this model. You need to use a jsonl file as the input. The only required arguments for this script are the path to test data and path to the directory in which you would like to save the results.
You can simply run the code with the following command:

```
python bert-lm.py -t path_to_jsonl_test_file -o path_to_results_directory
```