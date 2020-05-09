from utils import constant

def postprocess(preds):
    processed = []
    for ps in preds:
        new = []
        if len(ps) > 2 and ps[-1] != '.' and ps[-2] == '.':
            ps = ps[:-1]
        for i, p in enumerate(ps):
            if i > 0 and ps[i - 1] == p:
                continue
            new += [p]
        processed += [new]
    return processed


def save_predictions(preds, filename):
    with open(filename, 'w') as outfile:
        for tokens in preds:
            print(' '.join(tokens), file=outfile)
    print("Predictions saved to file: " + filename)


def unmap_with_copy(indices, src_tokens, vocab):
    result = []
    for ind, tokens in zip(indices, src_tokens):
        words = []
        for idx in ind:
            if idx >= 0:
                words.append(vocab.id2word[idx])
            else:
                idx = -idx - 1  # flip and minus 1
                words.append(tokens[idx])
        result += [words]
    return result


def prune_decoded_seqs(seqs):
    out = []
    for s in seqs:
        if constant.EOS_TOKEN in s:
            idx = s.index(constant.EOS_TOKEN)
            out += [s[:idx]]
        else:
            out += [s]
    return out


def unsort(sorted_list, oidx):
    _, unsorted = [list(t) for t in zip(*sorted(zip(oidx, sorted_list)))]
    return unsorted
