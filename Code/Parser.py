# coding=utf-8
# define the parser(arc-eager parser)
import copy
import random
from collections import defaultdict
import math
import numpy as np
from utils import *
from yy_flags import Flags
import pickle as pkl
class Configuration:
    def __init__(self, Buffer, discourse):
        self.discourse = discourse                      # the whole info of a discourse
        self.buffer = Buffer                            # the content of buffer
        self.stack = []                                 # the content of stack
        self.action_stack = []                          # action history information
        self.arcs = []                                  # the arcs that have been recognized

class GoldConfiguration:
    def __init__(self):
        self.heads = {}                                 # head node info
        self.deps = defaultdict(lambda: [])             # dependant not info

class Arc_Eager_Parser:
    SHIFT, RIGHT, LEFT, REDUCE = 0, 1, 2, 3
    transition2id = {"SHIFT": 0, 'RIGHT': 1, 'LEFT': 2, 'REDUCE': 3}
    if Flags.data_mode_setting=='UAS':
        action2id = pkl.load(open('./tmp_data/UAS_action2id.pkl', 'rb'))
    elif Flags.data_mode_setting=='LAS_Fine':
        action2id = pkl.load(open('./tmp_data/LAS_Fine_action2id.pkl', 'rb'))
    else:
        action2id = pkl.load(open('./tmp_data/LAS_Coarse_action2id.pkl', 'rb'))
    # action2id = pkl.load(open('./tmp_data/UAS_action2id.pkl', 'rb'))
    id2transition = {}
    id2action = {}
    for key in transition2id.keys():
        id2transition[transition2id[key]] = key
    for key in action2id.keys():
        id2action[action2id[key]] = key
    def __init__(self):
        self.transition_funcs = {}
        self.transition_funcs[Arc_Eager_Parser.SHIFT] = Arc_Eager_Parser.shift
        self.transition_funcs[Arc_Eager_Parser.RIGHT] = Arc_Eager_Parser.arc_right
        self.transition_funcs[Arc_Eager_Parser.LEFT] = Arc_Eager_Parser.arc_left
        self.transition_funcs[Arc_Eager_Parser.REDUCE] = Arc_Eager_Parser.reduce

    @staticmethod
    def terminal(conf):
        '''
        if the stack has no node and buffer only has a root node,
        we get the terminal state.(we only could do SHIFT action, and thus stack has the whole tree)
        '''
        return len(conf.stack) == 0 and len(conf.buffer) == 1

    @staticmethod
    def get_gold_conf(discourse):
        '''
        get the gold configuration based on the gold discourse info
        the head info of every edu node
        and the dependants of every edu node
        :param discourse: a discourse in train(validation, test) set
        :return: gold configuration
        '''
        gold_conf = GoldConfiguration()
        for dep in range(len(discourse)):
            head = discourse[dep][2]
            gold_conf.heads[dep] = head
            gold_conf.deps[head].append(dep)
        return gold_conf

    def test(self, discourse, sess, model):
        conf = self.initial(discourse)
        while not Arc_Eager_Parser.terminal(conf):
            scores = list(test_step(sess, [conf], model))[0]
            legal_transitions = self.legal(conf)
            t_p = 0  # default setting
            Relation = ""
            Max_score = 0
            First_in_legal = True
            for i in range(Flags.Class_size):
                temp_t_p = self.transition2id[self.id2action[i].split('_')[0]]
                temp_relation = self.id2action[i].split('_')[1]
                if temp_t_p in legal_transitions:
                    temp_score = scores[i]
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

    def get_gold_relation(self, conf, t_gold):
        if Flags.data_mode_setting=='UAS':
            return ""
        Relation = ""
        if t_gold == 1:  # Right
            Relation = conf.discourse[conf.buffer[0]][3]
        elif t_gold == 2:  # Left
            Relation = conf.discourse[conf.stack[-1]][3]
        return Relation

    def train(self, Discourse, sess, model, train_op):
        conf = self.initial(Discourse)
        gold_conf = Arc_Eager_Parser.get_gold_conf(Discourse)
        train_correct = train_all = 0.0
        ConfList, Gold_actionList = [], []
        while not Arc_Eager_Parser.terminal(conf):
            legal_transitions = self.legal(conf)
            gold_transitions = self.dyn_oracle(gold_conf, conf, legal_transitions)
            if len(gold_transitions) == 0:
                raise Exception('no gold_transition')
            t_gold = gold_transitions[0]
            Relation = self.get_gold_relation(conf, t_gold)
            temp_conf = copy.deepcopy(conf)
            ConfList.append(temp_conf)
            Gold_actionList.append(self.action2id[self.id2transition[t_gold] + '_' + Relation])
            conf = self.transition(t_gold, conf, Relation)

        # random shuffle the configuration list
        seed = 547
        random.seed(seed)
        random.shuffle(ConfList)
        random.seed(seed)
        random.shuffle(Gold_actionList)

        for i in range(int(math.ceil(len(ConfList) / 10.0))):
            scores = train_step(sess, ConfList[i * 10:(i + 1) * 10], Gold_actionList[i * 10:(i + 1) * 10], model, train_op)
            batch_gold_list = Gold_actionList[i * 10:(i + 1) * 10]
            for batch_index in range(len(batch_gold_list)):
                score = scores[batch_index]
                t_o = batch_gold_list[batch_index]
                t_p = np.argmax(score)
                if t_p == t_o:
                    train_correct += 1
                train_all += 1
        print('acc', train_correct/train_all)
        return train_correct, train_all

    def transition(self, t, conf, Relation):
        if Flags.action_history_mode=='no_relation':
            conf.action_stack.append(t)
        else:
            conf.action_stack.append(self.action2id[self.id2transition[t] + '_' + Relation])
        return self.transition_funcs[t](conf, Relation)


    def initial(self, Discourse):
        return Configuration(list(range(len(Discourse))) + [len(Discourse)], Discourse)

    def legal(self, conf):
        transitions = [Arc_Eager_Parser.SHIFT, Arc_Eager_Parser.RIGHT,
                       Arc_Eager_Parser.LEFT, Arc_Eager_Parser.REDUCE]
        shift_ok, right_ok, left_ok, reduce_ok = True, True, True, True
        if len(conf.buffer) == 1:
            # if buffer only has Root EDU, then can not do RIGHT and SHIFT transition
            right_ok = shift_ok = False
        if len(conf.stack) == 0:
            # if stack has no EDU, then can not do LEFT, RIGHT and REDUCE transition
            left_ok = right_ok = reduce_ok = False
        else:
            s = conf.stack[-1]
            # if the s is already a dependent, we cannot left-arc
            if len(list(filter(lambda hd: s == hd[1], conf.arcs))) > 0:
                left_ok = False
            else:
                reduce_ok = False
        ok = [shift_ok, right_ok, left_ok, reduce_ok]
        legal_transitions = []
        for it in range(len(transitions)):
            if ok[it] is True:
                legal_transitions.append(it)
        return legal_transitions

    def dyn_oracle(self, gold_conf, conf, legal_transitions):
        options = []
        if Arc_Eager_Parser.SHIFT in legal_transitions and Arc_Eager_Parser.zero_cost_shift(conf, gold_conf):
            options.append(Arc_Eager_Parser.SHIFT)
        if Arc_Eager_Parser.RIGHT in legal_transitions and Arc_Eager_Parser.zero_cost_right(conf, gold_conf):
            options.append(Arc_Eager_Parser.RIGHT)
        if Arc_Eager_Parser.LEFT in legal_transitions and Arc_Eager_Parser.zero_cost_left(conf, gold_conf):
            options.append(Arc_Eager_Parser.LEFT)
        if Arc_Eager_Parser.REDUCE in legal_transitions and Arc_Eager_Parser.zero_cost_reduce(conf, gold_conf):
            options.append(Arc_Eager_Parser.REDUCE)
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
        for bi in range(b, len(conf.discourse) + 1):
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

def train_step(sess, Conf_List, Gold_action_List, model, train_op):
    S_word_lists, B_word_lists, A_lists = [], [], []
    S_pos_lists, B_pos_lists = [], []
    S_sent_num, B_sent_num = [], []  # sentence num information
    S_duan_num, B_duan_num = [], []  # duan num information
    S_text_num, B_text_num = [], []  # discourse num information
    Same_lists, Induan_dis_lists, = [], []
    for conf in Conf_List:
        S_vector_list = get_stack_vector_list(conf.stack, conf.discourse, Flags.Stack_edus)
        S_word_lists.append(S_vector_list[0])
        S_pos_lists.append(S_vector_list[1])
        S_sent_num.append(S_vector_list[2])
        S_duan_num.append(S_vector_list[3])
        S_text_num.append(S_vector_list[4])

        B_vector_list = get_buffer_vector_list(conf.buffer, conf.discourse, Flags.Buffer_edus)
        B_word_lists.append(B_vector_list[0])
        B_pos_lists.append(B_vector_list[1])
        B_sent_num.append(B_vector_list[2])
        B_duan_num.append(B_vector_list[3])
        B_text_num.append(B_vector_list[4])

        Same_vector = get_same_vector(conf.stack, conf.buffer, conf.discourse)
        Same_lists.append(Same_vector)

        Induan_dis = get_induan_dis(conf.stack, conf.buffer, conf.discourse)
        Induan_dis_lists.append(Induan_dis)

        A_vector_list = get_action_vector_list(conf.action_stack, Flags.Action_actions)
        A_lists.append(A_vector_list)
    gold_action = [[1 if i == t_o else 0 for i in range(Flags.Class_size)] for t_o in Gold_action_List]
    feed_dict = {}
    feed_dict[model.input_S_word] = S_word_lists
    feed_dict[model.input_S_POS] = S_pos_lists
    feed_dict[model.input_B_word] = B_word_lists
    feed_dict[model.input_B_POS] = B_pos_lists
    feed_dict[model.input_A_his] = A_lists
    if Flags.neural_model_setting!='Basic_model':
        feed_dict[model.input_S_sent] = S_sent_num
        feed_dict[model.input_S_para] = S_duan_num
        feed_dict[model.input_S_disc] = S_text_num
        feed_dict[model.input_B_sent] = B_sent_num
        feed_dict[model.input_B_para] = B_duan_num
        feed_dict[model.input_B_disc] = B_text_num
        feed_dict[model.input_same] = Same_lists
        feed_dict[model.input_para_dis] = Induan_dis_lists

    feed_dict[model.gold_action] = gold_action

    predict_scores, _,en_loss, = sess.run([model.prob, train_op, model.en_loss], feed_dict)
    # print("entropy_loss", en_loss)
    return predict_scores


def test_step(sess, Conf_List, model):
    S_word_lists, B_word_lists, A_lists = [], [], []
    S_pos_lists, B_pos_lists = [], []
    S_sent_num, B_sent_num = [], []  # sentence num information
    S_duan_num, B_duan_num = [], []  # duan num information
    S_text_num, B_text_num = [], []  # discourse num information
    Same_lists, Induan_dis_lists, = [], []
    for conf in Conf_List:
        S_vector_list = get_stack_vector_list(conf.stack, conf.discourse, Flags.Stack_edus)
        S_word_lists.append(S_vector_list[0])
        S_pos_lists.append(S_vector_list[1])
        S_sent_num.append(S_vector_list[2])
        S_duan_num.append(S_vector_list[3])
        S_text_num.append(S_vector_list[4])

        B_vector_list = get_buffer_vector_list(conf.buffer, conf.discourse, Flags.Buffer_edus)
        B_word_lists.append(B_vector_list[0])
        B_pos_lists.append(B_vector_list[1])
        B_sent_num.append(B_vector_list[2])
        B_duan_num.append(B_vector_list[3])
        B_text_num.append(B_vector_list[4])

        Same_vector = get_same_vector(conf.stack, conf.buffer, conf.discourse)
        Same_lists.append(Same_vector)

        Induan_dis = get_induan_dis(conf.stack, conf.buffer, conf.discourse)
        Induan_dis_lists.append(Induan_dis)

        A_vector_list = get_action_vector_list(conf.action_stack, Flags.Action_actions)
        A_lists.append(A_vector_list)
    feed_dict = {}
    feed_dict[model.input_S_word] = S_word_lists
    feed_dict[model.input_S_POS] = S_pos_lists
    feed_dict[model.input_B_word] = B_word_lists
    feed_dict[model.input_B_POS] = B_pos_lists
    feed_dict[model.input_A_his] = A_lists
    if Flags.neural_model_setting!='Basic_model':
        feed_dict[model.input_S_sent] = S_sent_num
        feed_dict[model.input_S_para] = S_duan_num
        feed_dict[model.input_S_disc] = S_text_num
        feed_dict[model.input_B_sent] = B_sent_num
        feed_dict[model.input_B_para] = B_duan_num
        feed_dict[model.input_B_disc] = B_text_num
        feed_dict[model.input_same] = Same_lists
        feed_dict[model.input_para_dis] = Induan_dis_lists
    predict_scores = sess.run(model.prob, feed_dict)
    return predict_scores

