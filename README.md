## ACL2018 Modeling discourse cohesion for parsing via memory network

**This repository is the source code of ACL2018 short paper "Modeling discourse cohesion for parsing via memory network"**

In discourse parsing, how to model the difficult long span relationship is crucial. To solve this problem, most existing approaches design complicated features or exploit various off-the-shelf tools, but achieve little success. In this paper, we propose **a novel transition-based parsing model which makes use of memory network to capture discourse cohesion**. The automatically captured discourse cohesion benefits discourse parsing, especially for long spans. Experiments on **RST discourse treebank** show that our method outperforms traditional featured based methods and the memory based discourse cohesion can improve parsing performance significantly.

### Codes

***

The source codes are in **Code** directory, **parse.py** is the main function. For each setting of our model, you just need get in the exact directory and command **python parse.py** to run it.


### Requirements

***
- Python (=r3.5)
- Tensorflow (=r1.0)
- tflearn (=r0.3.2)

### Input illustration

***
We adopt RST corpus to perform the experiments. We give the input format description and one sample discourse in test set below:

Format: 

ID|EDU|part-of-speech tags of the words in the EDU|Head ID|Relation|Sentence ID|Position of EDU in Sentence|Position of EDU in discourse|paragraph ID|Position of EDU in paragraph|EDU ID in paragraph|X|X|

Sample discourse:

  1|Bruce W. Wilkinson , president and chief executive officer , was named to the additional post of chairman of this architectural and design services concern .|NN NN NN , NN CC JJ JJ NN , VBD VBN TO DT JJ NN IN NN IN DT JJ CC NN NNS NN .|0|Root|1|1|1|1|1|1|1|27

  2|Mr. Wilkinson , 93 years old , succeeds Thomas A. Bullock , 93 ,|NNP NNP , CD NNS JJ , VBZ IN NNP NN , CD ,|1|Background|2|1|2|1|2|2|1|16

  3|who is retiring as chairman|WP VBZ VBG IN NN|4|Contrast|2|2|3|1|3|3|2|5

  4|but will continue as a director and chairman of the executive committee .|CC MD VB IN DT NN CC NN IN DT NN NN .|2|Elaboration|2|3|3|1|3|4|3|13
