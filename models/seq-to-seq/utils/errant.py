"""
Scoring with ERRANT metric
"""
import subprocess
import os

def get_errant(hypotheses, reference):
    assert len(hypotheses) == len(reference)
    assert len(hypotheses) > 0
    with open('hyp1.txt', mode='w') as f:
        for h in hypotheses:
            f.write(h)
            f.write('\n')

    with open('ref.txt', mode='w') as f:
        for r in reference:
            f.write(r)
            f.write('\n')
    subprocess.call(["errant_parallel", "-orig", "hyp1.txt", "-cor", "ref.txt", "-out", "hyp.val_m2"])
    p = subprocess.Popen(["errant_compare", "-hyp", "hyp.val_m2", "-ref", "dataset/GEC/val_m2/ABCN.dev.gold.bea19.m2"], stdout=subprocess.PIPE)
    res = p.communicate()
    import pdb;pdb.set_trace()
    numbers = res[0].split(b'\t')[-3:]
    numbers = [float(n.split(b'\n')[0].decode("utf-8")) for n in numbers]
    return numbers[0], numbers[1], numbers[2]
