from __future__ import division
import torch
from utils import constant

"""

I adapted this from Open-NMT package...

"""


class Beam(object):
    def __init__(self, size, cuda=False):

        self.size = size
        self.done = False

        self.tt = torch.cuda if cuda else torch

        self.scores = self.tt.FloatTensor(size).zero_()
        self.allScores = []

        self.prevKs = []

        self.nextYs = [self.tt.LongTensor(size).fill_(constant.PAD_ID)]
        self.nextYs[0][0] = constant.SOS_ID

        self.copy = []

    def get_current_state(self):
        return self.nextYs[-1]

    def get_current_origin(self):
        return self.prevKs[-1]

    def advance(self, wordLk, copy_indices=None):
        if self.done:
            return True
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)
        else:
            beamLk = wordLk[0]

        flatBeamLk = beamLk.view(-1)

        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)
        self.allScores.append(self.scores)
        self.scores = bestScores

        prevK = bestScoresId / numWords
        self.prevKs.append(prevK)
        self.nextYs.append(bestScoresId - prevK * numWords)
        if copy_indices is not None:
            self.copy.append(copy_indices.index_select(0, prevK))

        if self.nextYs[-1][0] == constant.EOS_ID:
            self.done = True
            self.allScores.append(self.scores)

        return self.done

    def sort_best(self):
        return torch.sort(self.scores, 0, True)

    def get_best(self):
        scores, ids = self.sortBest()
        return scores[1], ids[1]

    def get_hyp(self, k):
        hyp = []
        cpy = []
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k])
            if len(self.copy) > 0:
                cpy.append(self.copy[j][k])
            k = self.prevKs[j][k]

        hyp = hyp[::-1]
        cpy = cpy[::-1]
        for i, cidx in enumerate(cpy):
            if cidx >= 0:
                hyp[i] = -(cidx + 1)

        return hyp
