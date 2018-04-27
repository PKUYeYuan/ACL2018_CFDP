# _*_coding=utf-8_*_
"""
通过Stack， Buffer，ActionStack的状态获得他们的向量表示
"""
from word_embedding import word_to_index

pos_num = 45
pos_dict = {'RBR': 37, 'JJR': 18, '-LRB-': 19, 'VBG': 23, 'CD': 3,
            'UH': 43, 'NNS': 8, 'CC': 6, 'WP': 34, '.': 13, 'VBN': 16,
            "''": 33, 'LS': 42, 'NNP': 1, 'PDT': 38, 'VBD': 22, 'IN': 15,
            'PRP$': 29, '$': 17, 'JJS': 30, '-RRB-': 20, 'NN': 7, '``': 36,
            ':': 14, 'WDT': 26, 'FW': 25, 'TO': 11, 'WP$': 41, 'POS': 24,
            'VBZ': 21, ',': 2, 'MD': 27, 'VBP': 10, 'JJ': 5, 'RB': 9, 'EX': 39,
            '#': 44, 'PRP': 31, 'RBS': 35, 'DT': 4, 'WRB': 32, 'SYM': 40, 'RP': 28, 'VB': 12}


def change_edu_to_index_vector(i, Artical):
    word_vector = []
    pos_vector = []
    # 得到相关的特征信息 edu word, edu pos
    edu = Artical[i][0]
    edupos = Artical[i][1]
    edu_sentence_info = Artical[i][5]
    edu_duan_info = Artical[i][8]
    edu_text_info = Artical[i][6]
    words = edu.split()
    poses = edupos.split()
    # word
    for word in words:
        if word in word_to_index.keys():
            word_vector.append(word_to_index[word])
        else:
            word_vector.append(0)
    # pos
    for pos in poses:
        if pos in pos_dict.keys():
            pos_vector.append(pos_dict[pos])
        else:
            pos_vector.append(0)

    return word_vector, pos_vector, edu_sentence_info, edu_duan_info, edu_text_info


def get_stack_vector_list(Stack, artical, handle_len):
    word, pos, sent_num, duan_num, text_num = [], [], [], [], []
    stack_length = len(Stack)
    if stack_length < handle_len:
        padding_length = handle_len - stack_length
        for i in range(padding_length):
            word.append([])
            pos.append([])
            sent_num.append(0)
            duan_num.append(0)
            text_num.append(0)
        for i in range(stack_length):
            temp_result = change_edu_to_index_vector(Stack[i], artical)
            word.append(temp_result[0])
            pos.append(temp_result[1])
            sent_num.append(temp_result[2])
            duan_num.append(temp_result[3])
            text_num.append(temp_result[4])
    else:
        for i in range(handle_len):
            temp_result = change_edu_to_index_vector(Stack[stack_length - handle_len + i], artical)
            word.append(temp_result[0])
            pos.append(temp_result[1])
            sent_num.append(temp_result[2])
            duan_num.append(temp_result[3])
            text_num.append(temp_result[4])
    return word, pos, sent_num, duan_num, text_num


def get_buffer_vector_list(Buffer, artical, handle_len):
    word, pos, sent_num, duan_num, text_num = [], [], [], [], []
    buffer_length = len(Buffer) - 1
    if buffer_length < handle_len:
        padding_length = handle_len - buffer_length
        for i in range(padding_length):
            word.append([])
            pos.append([])
            sent_num.append(0)
            duan_num.append(0)
            text_num.append(0)
        for i in range(buffer_length):
            temp_result = change_edu_to_index_vector(Buffer[buffer_length - 1 - i], artical)
            word.append(temp_result[0])
            pos.append(temp_result[1])
            sent_num.append(temp_result[2])
            duan_num.append(temp_result[3])
            text_num.append(temp_result[4])
    else:
        for i in range(handle_len):
            temp_result = change_edu_to_index_vector(Buffer[handle_len - 1 - i], artical)
            word.append(temp_result[0])
            pos.append(temp_result[1])
            sent_num.append(temp_result[2])
            duan_num.append(temp_result[3])
            text_num.append(temp_result[4])
    return word, pos, sent_num, duan_num, text_num


def get_same_vector(stack, Buffer, artical):  # 是否在一句  是否在一段
    Same_vector = [0, 0, 0, 0]
    if len(stack) == 0:
        return Same_vector
    if len(Buffer) == 1:
        return Same_vector
    if (artical[Buffer[0]][10] - artical[stack[-1]][10]) == 0:  # 是否在同一句
        Same_vector[0] = 1
    else:
        Same_vector[1] = 1
    if (artical[Buffer[0]][9] - artical[stack[-1]][9]) == 0:  # 是否在同一段
        Same_vector[2] = 1
    else:
        Same_vector[3] = 1
    return Same_vector


def get_induan_dis(stack, Buffer, artical):
    Empty = [0 for i in range(17)]
    if len(stack) == 0:
        return Empty
    if len(Buffer) == 1:
        return Empty
    if (artical[Buffer[0]][7] - artical[stack[-1]][7]) == 0:
        cha = [1 if i == artical[Buffer[0]][9] - artical[stack[-1]][9] + 5 else 0 for i in range(16)]
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
            action_vector.append(4)
        for i in range(action_length):
            action_vector.append(actionStack[i])
    else:
        for i in range(handle_len):
            action_vector.append(actionStack[action_length - handle_len + i])
    return action_vector
