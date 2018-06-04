# coding=utf-8
# some functions that help to pad, get position features
from yy_flags import Flags
"""
some functions to get the neural model input
"""
EDU_word = 0
EDU_POS = 1
head = 2
label = 3
sent_id = 4
EDU_pos_in_sent = 5
EDU_pos_in_disc = 6
parag_id = 7
EDU_pos_in_parag = 8
EDU_id_in_para = 9
def change_edu_to_index_vector(i, discourse):
    word_vector = []
    pos_vector = []
    word_vector = discourse[i][EDU_word]
    pos_vector = discourse[i][EDU_POS]
    edu_sent_info = discourse[i][EDU_pos_in_sent]
    edu_parag_info = discourse[i][EDU_pos_in_parag]
    edu_disc_info = discourse[i][EDU_pos_in_disc]

    return word_vector, pos_vector, edu_sent_info, edu_parag_info, edu_disc_info


def get_stack_vector_list(Stack, discourse, handle_len):
    word, pos, sent_num, duan_num, text_num = [], [], [], [], []
    stack_length = len(Stack)
    if stack_length < handle_len:
        padding_length = handle_len - stack_length
        for i in range(padding_length):
            word.append([0 for i in range(Flags.edu_padding_length)])
            pos.append([0 for i in range(Flags.edu_padding_length)])
            sent_num.append(0)
            duan_num.append(0)
            text_num.append(0)
        for i in range(stack_length):
            temp_result = change_edu_to_index_vector(Stack[i], discourse)
            word.append(temp_result[0])
            pos.append(temp_result[1])
            sent_num.append(temp_result[2])
            duan_num.append(temp_result[3])
            text_num.append(temp_result[4])
    else:
        for i in range(handle_len):
            temp_result = change_edu_to_index_vector(Stack[stack_length - handle_len + i], discourse)
            word.append(temp_result[0])
            pos.append(temp_result[1])
            sent_num.append(temp_result[2])
            duan_num.append(temp_result[3])
            text_num.append(temp_result[4])
    return word, pos, sent_num, duan_num, text_num


def get_buffer_vector_list(Buffer, discourse, handle_len):
    word, pos, sent_num, duan_num, text_num = [], [], [], [], []
    buffer_length = len(Buffer) - 1
    if buffer_length < handle_len:
        padding_length = handle_len - buffer_length
        for i in range(padding_length):
            word.append([0 for i in range(Flags.edu_padding_length)])
            pos.append([0 for i in range(Flags.edu_padding_length)])
            sent_num.append(0)
            duan_num.append(0)
            text_num.append(0)
        for i in range(buffer_length):
            temp_result = change_edu_to_index_vector(Buffer[buffer_length - 1 - i], discourse)
            word.append(temp_result[0])
            pos.append(temp_result[1])
            sent_num.append(temp_result[2])
            duan_num.append(temp_result[3])
            text_num.append(temp_result[4])
    else:
        for i in range(handle_len):
            temp_result = change_edu_to_index_vector(Buffer[handle_len - 1 - i], discourse)
            word.append(temp_result[0])
            pos.append(temp_result[1])
            sent_num.append(temp_result[2])
            duan_num.append(temp_result[3])
            text_num.append(temp_result[4])
    return word, pos, sent_num, duan_num, text_num


def get_same_vector(stack, Buffer, discourse):  # in the same sentence or paragraph
    Same_vector = [0, 0, 0, 0]
    if len(stack) == 0:
        return Same_vector
    if len(Buffer) == 1:
        return Same_vector
    if (discourse[Buffer[0]][sent_id] - discourse[stack[-1]][sent_id]) == 0:  # in the same sentence?
        Same_vector[0] = 1
    else:
        Same_vector[1] = 1
    if (discourse[Buffer[0]][parag_id] - discourse[stack[-1]][parag_id]) == 0:  # in the same paragraph?
        Same_vector[2] = 1
    else:
        Same_vector[3] = 1
    return Same_vector


def get_induan_dis(stack, Buffer, discourse):
    Empty = [0 for i in range(17)]
    if len(stack) == 0:
        return Empty
    if len(Buffer) == 1:
        return Empty
    if (discourse[Buffer[0]][parag_id] - discourse[stack[-1]][parag_id]) == 0:
        cha = [1 if i == discourse[Buffer[0]][EDU_id_in_para] - discourse[stack[-1]][EDU_id_in_para] + 5 else 0 for i in range(16)]
        if cha == [0 for i in range(16)]:
            cha.append(1)
        else:
            cha.append(0)
        return cha
    else:
        return [1 if i == 16 else 0 for i in range(17)]


def get_action_vector_list(actionStack, handle_len):
    action_length = len(actionStack)
    action_vector = []
    if action_length < handle_len:
        padding_length = handle_len - action_length
        for i in range(padding_length):
            # action_vector.append(Flags.Class_size)
            action_vector.append(Flags.action_history_kinds)
        for i in range(action_length):
            action_vector.append(actionStack[i])
    else:
        for i in range(handle_len):
            action_vector.append(actionStack[action_length - handle_len + i])
    return action_vector
