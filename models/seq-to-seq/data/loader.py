import random
import torch

from utils import constant, jsonl

class DataLoader(object):
    def __init__(self, filename, batch_size, opt, vocab, evaluation=False):
        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab
        self.eval = evaluation

        with open(filename) as infile:
            data = jsonl.load(infile)

        self.raw_data = data
        data = self.preprocess(data, vocab, opt)

        # shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
            self.raw_data = [self.raw_data[i] for i in indices]

        self.num_examples = len(data)

        # batching
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created for {}.".format(len(data), filename))

    def preprocess(self, data, vocab, opt):
        processed = []
        for d in data:
            # remove section headers (e.g., "FINDINGS :")
            src_tokens = d['src'][2:]
            tgt_tokens = d['tgt'][2:]
            optional_params = [constant.UNK_TOKEN]  # use unk as a dummy
            if opt['lower']:
                src_tokens = [t.lower() for t in src_tokens]
                tgt_tokens = [t.lower() for t in tgt_tokens]
                optional_params = [constant.UNK_TOKEN] # use unk as a dummy background
            src_tokens = [constant.SOS_TOKEN] + src_tokens + [constant.EOS_TOKEN]
            tgt_in = [constant.SOS_TOKEN] + tgt_tokens # target fed in RNN
            tgt_out = tgt_tokens + [constant.EOS_TOKEN] # target from RNN output
            src = map_to_ids(src_tokens, vocab.word2id)
            tgt_in = map_to_ids(tgt_in, vocab.word2id)
            tgt_out = map_to_ids(tgt_out, vocab.word2id)
            optional_args = map_to_ids(optional_params, vocab.word2id)
            processed += [[src_tokens, tgt_tokens, src, tgt_in, tgt_out, optional_args]]
        return processed

    def save_gold(self, filename):
        gold = [d['tgt'][2:] for d in self.raw_data]
        if self.opt['lower']:
            gold = [[t.lower() for t in g] for g in gold]
        if len(filename) > 0:
            with open(filename, 'w') as outfile:
                for seq in gold:
                    print(" ".join(seq), file=outfile)
        return gold

    def get_src(self):
        return [d['src'][2:] for d in self.raw_data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 6
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)
        src_tokens = batch[0]
        tgt_tokens = batch[1]
        src = get_long_tensor(batch[2], batch_size)
        tgt_in = get_long_tensor(batch[3], batch_size)
        tgt_out = get_long_tensor(batch[4], batch_size)
        optional_arg = get_long_tensor(batch[5], batch_size)
        return (src, tgt_in, tgt_out, optional_arg,
                src_tokens, tgt_tokens, orig_idx)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids

def get_long_tensor(tokens_list, batch_size, pad_id=constant.PAD_ID):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(pad_id)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens

def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]

