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

