#_*_coding=utf-8_*_
import csv
WORD = 0
POS = 1
HEAD = 2
LABEL = 3
Sentence_Num = 4
Sentence_percent_Num = 5
Discourse = 6


def read_conll_deps(f):

    artical = []

    with open(f) as infile:
        lines = infile.readlines()

        sentence = []

        for line in lines:
            row = line.strip().split('|')
            if line == '\n':
                sentence = [tok if tok[HEAD] is not -1 \
                            else (tok[WORD].lower(), tok[POS], len(sentence), \
                                  tok[LABEL], tok[Sentence_Num], \
                                  tok[Sentence_percent_Num], tok[Discourse],\
                                  tok[7], tok[8], tok[9], tok[10], tok[11]) for tok in sentence]
                artical.append(sentence)
                sentence = []
                continue
            sentence.append((row[1].lower(), row[2], int(row[3]) - 1,\
                             row[4], int(row[5]), int(row[6]), int(row[7]),\
                             int(row[8]), int(row[9]), int(row[10]), int(row[11]), int(row[12])))
    return artical
# read_conll_deps('../../../../Data/dev_input_coarse_pos')
