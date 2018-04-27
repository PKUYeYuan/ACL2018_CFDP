# _*_coding=utf-8_*_
# no_batch
from Get_Vector import *
from Neural_model import train_single, train_update, class_Size, edu_padding_length, Ac_Re_Dict, Reverse_Ac_Re_D
from tflearn.data_utils import pad_sequences


def NetGetScore(conflist):
    S_vector_lists, B_vector_lists, A_vector_lists = [], [], []
    S_pos_lists, B_pos_lists = [], []
    S_sent_num, B_sent_num = [], []  # sentence num information
    S_duan_num, B_duan_num = [], []  # duan num information
    S_text_num, B_text_num = [], []  # discourse num information
    Same_lists, Induan_dis_lists, = [], []
    for conf in conflist:
        S_vector_list = get_stack_vector_list(conf.stack, conf.artical, 1)
        S_vector_lists.append(
            pad_sequences(S_vector_list[0], maxlen=edu_padding_length,
                          padding='pre', truncating='post', value=0))
        S_pos_lists.append(
            pad_sequences(S_vector_list[1], maxlen=edu_padding_length,
                          padding='pre', truncating='post', value=0))
        S_sent_num.append(S_vector_list[2])
        S_duan_num.append(S_vector_list[3])
        S_text_num.append(S_vector_list[4])

        B_vector_list = get_buffer_vector_list(conf.buffer, conf.artical, 2)
        B_vector_lists.append(
            pad_sequences(B_vector_list[0], maxlen=edu_padding_length,
                          padding='pre', truncating='post', value=0))
        B_pos_lists.append(
            pad_sequences(B_vector_list[1], maxlen=edu_padding_length,
                          padding='pre', truncating='post', value=0))
        B_sent_num.append(B_vector_list[2])
        B_duan_num.append(B_vector_list[3])
        B_text_num.append(B_vector_list[4])

        Same_vector = get_same_vector(conf.stack, conf.buffer, conf.artical)
        Same_lists.append(Same_vector)

        Induan_dis = get_induan_dis(conf.stack, conf.buffer, conf.artical)
        Induan_dis_lists.append(Induan_dis)

        A_vector_list = get_action_vector_list(conf.action_stack, 3)
        A_vector_lists.append(A_vector_list)
    pred = train_single(S_vector_lists, S_pos_lists, S_sent_num, S_duan_num, S_text_num,
                        B_vector_lists, B_pos_lists, B_sent_num, B_duan_num, B_text_num,
                        Same_lists, Induan_dis_lists,
                        A_vector_lists, )
    return pred


def NetUpdate(conflist, t_os):
    S_vector_lists, B_vector_lists, A_vector_lists = [], [], []
    S_pos_lists, B_pos_lists = [], []
    S_sent_num, B_sent_num = [], []  # sentence num information
    S_duan_num, B_duan_num = [], []  # duan num information
    S_text_num, B_text_num = [], []  # discourse num information
    Same_lists, Induan_dis_lists, = [], []
    for conf in conflist:
        S_vector_list = get_stack_vector_list(conf.stack, conf.artical, 1)
        S_vector_lists.append(
            pad_sequences(S_vector_list[0], maxlen=edu_padding_length,
                          padding='pre', truncating='post', value=0))
        S_pos_lists.append(
            pad_sequences(S_vector_list[1], maxlen=edu_padding_length,
                          padding='pre', truncating='post', value=0))
        S_sent_num.append(S_vector_list[2])
        S_duan_num.append(S_vector_list[3])
        S_text_num.append(S_vector_list[4])

        B_vector_list = get_buffer_vector_list(conf.buffer, conf.artical, 2)
        B_vector_lists.append(
            pad_sequences(B_vector_list[0], maxlen=edu_padding_length,
                          padding='pre', truncating='post', value=0))
        B_pos_lists.append(
            pad_sequences(B_vector_list[1], maxlen=edu_padding_length,
                          padding='pre', truncating='post', value=0))
        B_sent_num.append(B_vector_list[2])
        B_duan_num.append(B_vector_list[3])
        B_text_num.append(B_vector_list[4])

        Same_vector = get_same_vector(conf.stack, conf.buffer, conf.artical)
        Same_lists.append(Same_vector)

        Induan_dis = get_induan_dis(conf.stack, conf.buffer, conf.artical)
        Induan_dis_lists.append(Induan_dis)

        A_vector_list = get_action_vector_list(conf.action_stack, 3)
        A_vector_lists.append(A_vector_list)
    gold_action = [[1 if i == Ac_Re_Dict[t_o] else 0 for i in range(class_Size)] for t_o in t_os]
    Pred = train_update(S_vector_lists, S_pos_lists, S_sent_num, S_duan_num, S_text_num,
                        B_vector_lists, B_pos_lists, B_sent_num, B_duan_num, B_text_num,
                        Same_lists, Induan_dis_lists,
                        A_vector_lists, gold_action)
    return Pred
