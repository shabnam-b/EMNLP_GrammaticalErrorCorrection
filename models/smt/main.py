# smt code for enlp project
# shira wein, 2020

from __future__ import division
from nltk.translate import AlignedSent
from nltk.translate.ibm2 import IBMModel2
from nltk.tokenize import wordpunct_tokenize
import json

bitext = []

with open('train.json') as f:
    for line in f:
        curr = json.loads(line)
        #tokenizing source and target sentences for each line of json training file
        source = wordpunct_tokenize(curr['src'])
        #print("source: ", source)
        target = wordpunct_tokenize(curr['tgt'])
        #print("target: ", target)
        #creating bitext alignment for source and target sentences
        bitext.append(AlignedSent(source,target))

    f.close()

ibm2 = IBMModel2(bitext, 5)

#reference and test sets for evaluation
referenceset = set()
testset = set()

count = 0;

with open('dev.json') as d:
    for line2 in d:
        curr2 = json.loads(line2)
        #tokenizing source and target sentences for each line of json dev file
        source2 = wordpunct_tokenize(curr2['src'])
        #print("source sentence: ", source2)
        target2 = str(count)
        target2 += " "
        target2 += curr2['tgt']
        #adding target sentence to reference set
        referenceset.add(target2)
        #print("reference sentence: ", target2)

        result_sentence = str(count)
        for t in range(0, len(source2)):
            val = 0.0
            key = ""
            #finding most probable translation for the source token
            for k, v in ibm2.translation_table[source2[t]].items():
                if v >= val:
                    val = v
                    key = k
            #if key is "", add source token
            if(key is ""):
                result_sentence += " "
                result_sentence += source2[t]
            #if key (most probable token) is not None, then add the key to the sentence
            #None alignment means the token is most probably removed from the sentence
            if(key is not None):
                result_sentence += " "
                result_sentence += key
            #otherwise just add source token
            else:
                result_sentence += " "
                result_sentence += source2[t]

        #add resulting sentence to test set
        testset.add(result_sentence)
        #print("output sentence: ", result_sentence)
        print(result_sentence)

        #for nltk statistics, needed to add count token before printing output
        count += 1


    d.close()
