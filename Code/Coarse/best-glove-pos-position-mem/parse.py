# _*_coding=utf-8_*_
import copy
import random
from collections import defaultdict

from Neural_interface import NetGetScore, NetUpdate
from Neural_model import train_model_save, train_model_import, Ac_Re_Dict, Reverse_Ac_Re_D
import math

class Configuration:
    def __init__(self, Buffer, Artical):
        self.artical = Artical  # 文章的句子的信息
        self.buffer = Buffer  # 记录Buffer的信息
        self.stack = []  # 记录Stack的信息
        self.action_stack = []  # Action History信息
        self.arcs = []  # 记录边的信息


class GoldConfiguration:
    def __init__(self):
        self.heads = {}  # 节点的头节点信息
        self.deps = defaultdict(lambda: [])  # 节点的子节点信息


class GreedyDepParser:
    SHIFT, RIGHT, LEFT, REDUCE = 0, 1, 2, 3
    LUT = ["SHIFT", 'RIGHT', 'LEFT', 'REDUCE']
    TUL = {"SHIFT": 0, 'RIGHT': 1, 'LEFT': 2, 'REDUCE': 3}

    def __init__(self):
        self.transition_funcs = {}

    @staticmethod
    def terminal(conf):
        return len(conf.stack) == 0 and len(conf.buffer) == 1

    @staticmethod
    def get_gold_conf(Artical):
        gold_conf = GoldConfiguration()
        for dep in range(len(Artical)):
            head = Artical[dep][2]
            gold_conf.heads[dep] = head
            gold_conf.deps[head].append(dep)
        return gold_conf

    # We need to have arcs that are dominated with 
    # no crossing lines, excluding the root
    @staticmethod
    def non_projective(conf):
        for dep1 in conf.heads.keys():
            head1 = conf.heads[dep1]
            for dep2 in conf.heads.keys():
                head2 = conf.heads[dep2]
                if head1 < 0 or head2 < 0:
                    continue
                if (dep1 > head2 and dep1 < dep2 and head1 < head2) or (dep1 < head2 and dep1 > dep2 and head1 < dep2):
                    return True

                if dep1 < head1 and head1 is not head2:
                    if (head1 > head2 and head1 < dep2 and dep1 < head2) or (
                                head1 < head2 and head1 > dep2 and dep1 < dep2):
                        return True
        return False

    def legal(self, conf, do_reduce):
        pass

    def initial(self, Artical):
        pass

    def dyn_oracle(self, gold_conf, conf, legal_transitions):
        pass

    def run(self, Artical):
        conf = self.initial(Artical)
        while not GreedyDepParser.terminal(conf):
            scores = list(NetGetScore([conf]))[0]
            legal_transitions = self.legal(conf, True)
            t_p = 0  # default setting
            Relation = ""
            Max_score = 0
            First_in_legal = True
            for Ac_Re in Ac_Re_Dict:
                temp_t_p = self.TUL[Ac_Re.split('_')[0]]
                temp_relation = Ac_Re.split('_')[1]
                if temp_t_p in legal_transitions:
                    temp_score = scores[Ac_Re_Dict[Ac_Re]]
                    if temp_t_p == 2 and temp_relation == 'Root' and len(conf.buffer) > 1:
                        continue
                    if First_in_legal:
                        Max_score = temp_score
                        t_p = temp_t_p
                        Relation = temp_relation
                        First_in_legal = False
                    else:
                        if Max_score < temp_score:
                            Max_score = temp_score
                            t_p = temp_t_p
                            Relation = temp_relation
            conf = self.transition(t_p, conf, Relation)
        return conf.arcs

    @staticmethod
    def Can_Do_Reduce(conf, gold_conf):
        if len(conf.stack) == 0:
            return False
        S = conf.stack[-1]
        B = conf.buffer[0]
        Deps = gold_conf.deps[S]
        for Dep in Deps:
            if Dep > B:
                return False
        return True

    def get_gold_relation(self, conf, t_gold):
        Relation = ""
        if t_gold == 1:  # Right
            Relation = conf.artical[conf.buffer[0]][3]
        elif t_gold == 2:  # Left
            Relation = conf.artical[conf.stack[-1]][3]
        return Relation

    def train(self, Artical, iter_num):
        conf = self.initial(Artical)
        gold_conf = GreedyDepParser.get_gold_conf(Artical)
        train_correct = train_all = 0
        ConfList, Gold_actionList = [], []
        # batch的方式：句子，篇章？
        while not GreedyDepParser.terminal(conf):
            do_reduce = self.Can_Do_Reduce(conf, gold_conf)
            legal_transitions = self.legal(conf, do_reduce)
            gold_transitions = self.dyn_oracle(gold_conf, conf, legal_transitions)
            if len(gold_transitions) == 0:
                raise Exception('no gold_transition')
            t_gold = gold_transitions[0]
            Relation = self.get_gold_relation(conf, t_gold)
            temp_conf = copy.deepcopy(conf)
            ConfList.append(temp_conf)
            Gold_actionList.append(self.LUT[t_gold] + '_' + Relation)
            conf = self.transition(t_gold, conf, Relation)
        seed = 547
        random.seed(seed)
        random.shuffle(ConfList)
        random.seed(seed)
        random.shuffle(Gold_actionList)

        # 最后把连接到root这一过程的所有transition一起预测 #这里在buffer只剩root之后应该根据arcs和stack的情况，自行确定
        for i in range(int(math.ceil(len(ConfList)/10.0))):
            scores = NetUpdate(ConfList[i*10:(i+1)*10], Gold_actionList[i*10:(i+1)*10])
            temp_gold_list = Gold_actionList[i*10:(i+1)*10]
            for scores_iter in range(len(scores)):
                score = scores[scores_iter]
                t_o = temp_gold_list[scores_iter]
                t_p = Reverse_Ac_Re_D[max(Reverse_Ac_Re_D.keys(), key=lambda p: score[p])]
                if t_p == t_o:
                    train_correct += 1
                train_all += 1
        return train_correct, train_all

    def transition(self, t, conf, Relation):
        conf.action_stack.append(t)
        return self.transition_funcs[t](conf, Relation)


class ArcEagerDepParser(GreedyDepParser):
    def __init__(self):
        GreedyDepParser.__init__(self)
        self.transition_funcs[ArcEagerDepParser.SHIFT] = ArcEagerDepParser.shift
        self.transition_funcs[ArcEagerDepParser.RIGHT] = ArcEagerDepParser.arc_right
        self.transition_funcs[ArcEagerDepParser.LEFT] = ArcEagerDepParser.arc_left
        self.transition_funcs[ArcEagerDepParser.REDUCE] = ArcEagerDepParser.reduce

    def initial(self, Artical):
        return Configuration(list(range(len(Artical))) + [len(Artical)], Artical)

    def legal(self, conf, do_reduce):
        transitions = [GreedyDepParser.SHIFT, GreedyDepParser.RIGHT, \
                       GreedyDepParser.LEFT, GreedyDepParser.REDUCE]
        shift_ok, right_ok, left_ok, reduce_ok = True, True, True, True
        if len(conf.buffer) == 1:
            right_ok = shift_ok = False
        if len(conf.stack) == 0:
            left_ok = right_ok = reduce_ok = False
        else:
            s = conf.stack[-1]
            # if the s is already a dependent, we cannot left-arc
            if len(list(filter(lambda hd: s == hd[1], conf.arcs))) > 0:
                left_ok = False
            else:
                reduce_ok = False
        if not do_reduce:
            reduce_ok = False
        ok = [shift_ok, right_ok, left_ok, reduce_ok]
        legal_transitions = []
        for it in range(len(transitions)):
            if ok[it] is True:
                legal_transitions.append(it)
        return legal_transitions

    def dyn_oracle(self, gold_conf, conf, legal_transitions):
        options = []
        if GreedyDepParser.SHIFT in legal_transitions and ArcEagerDepParser.zero_cost_shift(conf, gold_conf):
            options.append(GreedyDepParser.SHIFT)
        if GreedyDepParser.RIGHT in legal_transitions and ArcEagerDepParser.zero_cost_right(conf, gold_conf):
            options.append(GreedyDepParser.RIGHT)
        if GreedyDepParser.LEFT in legal_transitions and ArcEagerDepParser.zero_cost_left(conf, gold_conf):
            options.append(GreedyDepParser.LEFT)
        if GreedyDepParser.REDUCE in legal_transitions and ArcEagerDepParser.zero_cost_reduce(conf, gold_conf):
            options.append(GreedyDepParser.REDUCE)
        return options

    @staticmethod
    def zero_cost_shift(conf, gold_conf):
        if len(conf.buffer) <= 1:
            return False
        b = conf.buffer[0]

        for si in conf.stack:
            if gold_conf.heads[si] == b or (gold_conf.heads[b] == si):
                return False
        return True

    @staticmethod
    def zero_cost_right(conf, gold_conf):
        if len(conf.stack) == 0 or len(conf.buffer) <= 1:
            return False
        s = conf.stack[-1]
        b = conf.buffer[0]
        if gold_conf.heads[b] == s:
            return True
        return False

    @staticmethod
    def zero_cost_left(conf, gold_conf):
        if len(conf.stack) == 0 or len(conf.buffer) < 1:
            return False
        s = conf.stack[-1]
        b = conf.buffer[0]
        if gold_conf.heads[s] == b:
            return True
        return False

    @staticmethod
    def zero_cost_reduce(conf, gold_conf):
        if len(conf.stack) is 0:
            return False
        s = conf.stack[-1]
        b = conf.buffer[0]
        for bi in range(b, len(conf.artical) + 1):
            if bi in gold_conf.heads and gold_conf.heads[bi] == s:
                return False
        return True

    @staticmethod
    def shift(conf, Relation):
        b = conf.buffer[0]
        del conf.buffer[0]
        conf.stack.append(b)
        return conf

    @staticmethod
    def arc_right(conf, Relation):
        s = conf.stack[-1]
        b = conf.buffer[0]
        del conf.buffer[0]
        conf.stack.append(b)
        conf.arcs.append((s, b, Relation))
        return conf

    @staticmethod
    def arc_left(conf, Relation):
        s = conf.stack.pop()
        b = conf.buffer[0]
        if len(conf.buffer) == 1:
            Relation = 'Root'
        conf.arcs.append((b, s, Relation))
        return conf

    @staticmethod
    def reduce(conf, Relation):
        conf.stack.pop()
        return conf


if __name__ == '__main__':
    import argparse
    import fileio

    parser = argparse.ArgumentParser(description="Sample program showing training and testing dependency parsers")
    parser.add_argument('--parser', help='Parser type (eager|hybrid) (default: eager)', default='eager')
    parser.add_argument('--train', help='CONLL training file', default='../../../Data/train_input_coarse_pos')
    parser.add_argument('--test', help='CONLL testing file', default='../../../Data/test_input_coarse_pos')
    parser.add_argument('--dev', help='CONLL testing file', default='../../../Data/dev_input_coarse_pos')
    parser.add_argument('--fx', help='Feature extractor', default='ex')
    parser.add_argument('--n', help='Number of passes over training data', default=5, type=int)
    parser.add_argument('-v', default=True)
    opts = parser.parse_args()


    def filter_non_projective(gold):
        gold_proj = []
        for s in gold:
            gold_conf = GreedyDepParser.get_gold_conf(s)
            if GreedyDepParser.non_projective(gold_conf) is False:
                gold_proj.append(s)
            elif opts.v is True:
                print('Skipping non-projective sentence', s)
        return gold_proj


    # Defaults
    Parser = ArcEagerDepParser
    parser = Parser()


    def dev(Iter):
        print("Dev Part Begin")
        train_model_import(Iter)
        dev = filter_non_projective(fileio.read_conll_deps(opts.dev))
        all_arcs, uas_correct, las_correct = 0.0, 0.0, 0.0

        for gold_dev_sent in dev:  # gold_dev_sent  每个篇章
            gold_arcs = set([(gold_dev_sent[i][2], i, gold_dev_sent[i][3]) for i in range(len(gold_dev_sent))])
            gold_arcs = sorted(gold_arcs, key=lambda x: x[1])
            arcs = set(parser.run(gold_dev_sent))
            temp_arcs = sorted(arcs, key=lambda x: x[1])
            for iter_i in range(len(gold_arcs)):
                if (temp_arcs[iter_i][0] == gold_arcs[iter_i][0]):
                    uas_correct += 1
                if (temp_arcs[iter_i][0] == gold_arcs[iter_i][0] and temp_arcs[iter_i][2] == gold_arcs[iter_i][2]):
                    las_correct += 1
            all_arcs += len(gold_arcs)

        print('uas accuracy %d/%d = %f' % (uas_correct, all_arcs, float(uas_correct) / float(all_arcs)))
        print('las accuracy %d/%d = %f' % (las_correct, all_arcs, float(las_correct) / float(all_arcs)))
        #         return float(uas_correct)/float(all_arcs)
        return float(las_correct) / float(all_arcs)

    def test(Iter):
        print("Test Part Begin")
        train_model_import(Iter)
        Test = filter_non_projective(fileio.read_conll_deps(opts.test))
        all_arcs, uas_correct, las_correct = 0.0, 0.0, 0.0

        testOutfile = open("./Result/" + "test" + str(Iter) + ".txt", 'a+')

        def Write_testOut(sents, arcs):  # 篇章和预测的head,arc
            temp_arcs = sorted(arcs, key=lambda x: x[1])
            sents_length = len(sents)
            for sent_iter in range(sents_length):
                sent = sents[sent_iter]
                outputStr = str(sent_iter + 1) + '|' + sent[0] + '|'
                if temp_arcs[sent_iter][1] == sent_iter:
                    if temp_arcs[sent_iter][0] == sents_length:
                        outputStr += '0|Root' + '\n'
                    else:
                        outputStr += str(temp_arcs[sent_iter][0] + 1) + '|' + temp_arcs[sent_iter][2] + '\n'
                    testOutfile.write(outputStr)
                else:
                    testOutfile.write("didn't predict this sentence!ERROR!\n")
            testOutfile.write('\n')

        for gold_test_sent in Test:
            gold_arcs = set([(gold_test_sent[i][2], i, gold_test_sent[i][3]) for i in range(len(gold_test_sent))])
            gold_arcs = sorted(gold_arcs, key=lambda x: x[1])
            arcs = set(parser.run(gold_test_sent))
            temp_arcs = sorted(arcs, key=lambda x: x[1])
            for iter_i in range(len(gold_arcs)):
                if (temp_arcs[iter_i][0] == gold_arcs[iter_i][0]):
                    uas_correct += 1
                if (temp_arcs[iter_i][0] == gold_arcs[iter_i][0] and temp_arcs[iter_i][2] == gold_arcs[iter_i][2]):
                    las_correct += 1
            all_arcs += len(gold_arcs)
            Write_testOut(gold_test_sent, arcs)
        print('uas accuracy %d/%d = %f' % (uas_correct, all_arcs, float(uas_correct) / float(all_arcs)))
        print('las accuracy %d/%d = %f' % (las_correct, all_arcs, float(las_correct) / float(all_arcs)))
        # Print_result_test()
        outfile = open("right_ratio.txt", 'a+')
        outfile.write("test: in the iteration " + str(Iter) + " the test uas right ratio is: " + str(
            float(uas_correct) / float(all_arcs)) + ' correct_arcs:' + str(uas_correct) + ' all_arcs:' + str(
            all_arcs) + '\n')
        outfile.write("test: in the iteration " + str(Iter) + " the test las right ratio is: " + str(
            float(las_correct) / float(all_arcs)) + ' correct_las:' + str(las_correct) + ' all_arcs:' + str(
            all_arcs) + '\n')
        outfile.close()
        testOutfile.close()
        return float(uas_correct) / float(all_arcs)

    def train():
        train_temp = fileio.read_conll_deps(opts.train)
        print("train_artical:", len(train_temp))
        Train = filter_non_projective(train_temp)
        print("Train Part Begin")
        decline_count, Iter = 0, 0
        max_right_ratio, right_ratio = 0.0, 0.0

        while (decline_count < 10):
            Iter += 1
            print("Iter: ", Iter, "Begin")
            if (right_ratio <= max_right_ratio):
                decline_count += 1
                if (Iter > 1):
                    test(Iter - 1)  # 加入的
            else:
                decline_count = 0
                max_right_ratio = right_ratio
                test(Iter - 1)
            correct_iter, all_iter = 0, 0
            random.shuffle(Train)
            artical_index = 0
            for gold_train_sent in Train:
                print("handle artical :" + str(artical_index))
                artical_index += 1
                correct_s, all_s = parser.train(gold_train_sent, Iter)
                correct_iter += correct_s
                all_iter += all_s
            print('fraction of correct transitions iteration %d: %d/%d = %f' % (
            Iter, correct_iter, all_iter, correct_iter / float(all_iter)))
            outfile = open("right_ratio.txt", 'a+')
            outfile.write("train: in the iteration " + str(Iter) + " the train right ratio is: " + str(
                correct_iter / float(all_iter)) + ' correct_iter:' + str(correct_iter) + ' all_iter:' + str(
                all_iter) + '\n')
            train_model_save(Iter)
            right_ratio = dev(Iter)
            print ("in the iteration ", Iter, "dev right_ratio:", right_ratio)
            outfile.write("dev:" + str(Iter) + " the dev right ratio is: " + str(right_ratio) + '\n')
            outfile.close()
            #             break
        return Iter

    mIter = train()
    test(mIter)

