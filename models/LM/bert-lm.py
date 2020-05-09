import json
import copy
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM
import logging
import torch
import argparse

args = argparse.ArgumentParser(description='Program description.')
args.add_argument('-t', '--test', help='Path to test jsonl file', required=True)
args.add_argument('-tr', '--threshold', help='threshold', default=0.6)
args.add_argument('-n', '--n', help='Number of times to go over the sentence', default=3)
args.add_argument('-o', '--output', help='Path to the directory of the output file', required=True)
args = args.parse_args()

logging.basicConfig(level=logging.INFO)
PAD, MASK, CLS, SEP = '[PAD]', '[MASK]', '[CLS]', '[SEP]'
USE_GPU = 1
# Device configuration
device = torch.device('cuda' if (torch.cuda.is_available() and USE_GPU) else 'cpu')

pretrained_model = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(pretrained_model)
model = BertForMaskedLM.from_pretrained(pretrained_model)
model.eval()


def preprocess(path):
    data_x = []
    with open(path, 'r') as json_file:
        for line in json_file:
            data = json.loads(line)
            data_x.append(data['src'][2:])
    return data_x


def masking(x):
    res = []
    for ii in tqdm(range(len(x))):
        sent = copy.deepcopy(x[ii])
        for jj in range(args.n):
            for kk in range(len(x[ii])):
                if sent[0] != CLS:
                    sent = [CLS] + sent
                if sent[-1] != SEP:
                    sent.append(SEP)
                org = copy.deepcopy(sent)
                sent[kk + 1] = MASK
                sent = predict(sent, org)
        res.append(sent[1:len(sent) - 1])

    return res


def predict(tokens, org):
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
    tokens_tensor = torch.tensor([indexed_tokens])
    mask_idx = tokens.index('[MASK]')
    segment_idx = tokens_tensor * 0
    tokens_tensor = tokens_tensor.to(device)
    segments_tensors = segment_idx.to(device)
    mask = (tokens_tensor != 103)
    mask.to(device)
    model.to(device)
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors, masked_lm_labels=None)
        outputs = outputs[0]
    probs = torch.softmax(outputs, dim=-1)[0]
    topk_prob, topk_indices = torch.topk(probs[mask_idx, :], 5)
    topk_tokens = tokenizer.convert_ids_to_tokens(topk_indices.cpu().numpy())
    if topk_prob[0] < args.threshold:
        return org
    for i in range(5):
        t = topk_tokens[i]
        if t != '[UNK]' and t != SEP and t != CLS and topk_prob[i] >= args.threshold:
            tokens[tokens.index('[MASK]')] = t
            return tokens
        else:
            return org


if __name__ == "__main__":
    dev_x = preprocess(args.test)
    res = masking(dev_x)

    with open(args.output + '/preds.json', 'w') as outfile:
        json.dump(res, outfile)
