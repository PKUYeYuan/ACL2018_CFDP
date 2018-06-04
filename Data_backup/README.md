Here is our sample data for training, validation and testing.

For each set(training, validation, testing), we use two kinds of data settings: Fine and Coarse,
which are different from the relation sets, Fine has 111 relations, Coarse has 19 relations.

In each file, every position in a line split by '|' means
ID|EDU|part-of-speech tags of the words in the EDU|Head ID|Relation|Sentence ID|Position of EDU in Sentence|Position of EDU in discourse|paragraph ID|Position of EDU in paragraph|EDU ID in paragraph|X|X|
,respectively. And every discourse is split by an empty line.

pos2id.txt describes the POS tags that appear in the training set.(using Stanford NLP tools to get the POS tags info)