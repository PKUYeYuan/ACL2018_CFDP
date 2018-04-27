# _*_coding=utf-8_*_
import tensorflow as tf
import os
from word_embedding import word_embedding_matrix
Ac_Re_Dict = {'RIGHT_Condition': 0, 'LEFT_Condition': 1, 'RIGHT_Attribution': 2, 'LEFT_Attribution': 3, 'LEFT_Summary': 4, 'RIGHT_Summary': 5,\
              'RIGHT_Elaboration': 6, 'LEFT_Elaboration': 7 ,\
              'LEFT_Explanation': 8, 'RIGHT_Explanation': 9, 'RIGHT_Enablement': 10, 'LEFT_Enablement': 11, \
              'LEFT_Same-Unit': 12, 'RIGHT_Same-Unit': 13, 'SHIFT_': 14, 'RIGHT_Joint': 15, 'LEFT_Joint': 16, \
              'RIGHT_Topic-Comment': 17, 'LEFT_Topic-Comment': 18, 'LEFT_Contrast': 19, 'RIGHT_Contrast': 20,\
              'LEFT_Temporal': 21, 'RIGHT_Temporal': 22, 'RIGHT_Evaluation': 23, 'LEFT_Evaluation': 24,\
              'RIGHT_Cause': 25, 'LEFT_Cause': 26, 'LEFT_Background': 27, 'RIGHT_Background': 28,\
              'LEFT_Topic-Change': 29, 'RIGHT_Topic-Change': 30, 'LEFT_Manner-Means': 31, 'RIGHT_Manner-Means': 32 ,\
              'LEFT_Comparison': 33, 'RIGHT_Comparison': 34, 'LEFT_TextualOrganization': 35, 'RIGHT_TextualOrganization': 36,\
              'LEFT_Span': 37,  'RIGHT_Span': 38, \
              'REDUCE_': 39, 'LEFT_Root':40}
Reverse_Ac_Re_D = {Ac_Re_Dict[S]:S for S in Ac_Re_Dict.keys()}
# 超参数设置
word_dim = 50  # 词向量维度
edu_rnn_dim = 200  # edu_embedding（针对word）的时候rnn输出向量维度

action_kinds = 5  # shift, left, right, reduce, null
action_dim = 50  # Action向量表示的维度

pos_kinds = 45  # pos数目 44kinds + 1padding
pos_dim = 15  # pos向量表示的维度
pos_rnn_dim = 50  # edu_embedding（针对pos）的时候rnn输出向量维度

max_position_num = 4  # 最大编号 1,2,3 + padding 0
position_dim = 15  # 编号信息维度

same_num = 4  # 表示是否在同一段,同一句中
same_fea_dim = 15  # 向量表示维度

dis_num = 17  # edu编号距离
dis_fea_dim = 15  # 距离表示的向量维度

mem_dim = 20  # mem slots
mem_beta = 10  # 一个常数

stack_edus = 1  # 从stack中取出的edu数目
buffer_edus = 2  # 从buffer中取出的edu数目
action_actions = 3  # 从action stack中取出的action数目
l2_lambda = 10**-5
class_Size = 41  # 分类器分类数目
edu_padding_length = 20  # edu单词数的padding长度
lr = 0.001  # 学习率
Globla_Step = 0
# 通用的参数，不会改变
stddev_setting = 0.01  # 初始化变量的时候标准差设置
'''模型部分'''
# 处理输入
input_S = tf.placeholder(dtype=tf.int32, shape=[None, stack_edus, edu_padding_length])
input_S_p = tf.placeholder(dtype=tf.int32, shape=[None, stack_edus, edu_padding_length])
input_S_sent = tf.placeholder(dtype=tf.int32, shape=[None, stack_edus])
input_S_duan = tf.placeholder(dtype=tf.int32, shape=[None, stack_edus])
input_S_text = tf.placeholder(dtype=tf.int32, shape=[None, stack_edus])

input_B = tf.placeholder(dtype=tf.int32, shape=[None, buffer_edus, edu_padding_length])
input_B_p = tf.placeholder(dtype=tf.int32, shape=[None, buffer_edus, edu_padding_length])
input_B_sent = tf.placeholder(dtype=tf.int32, shape=[None, buffer_edus])
input_B_duan = tf.placeholder(dtype=tf.int32, shape=[None, buffer_edus])
input_B_text = tf.placeholder(dtype=tf.int32, shape=[None, buffer_edus])

input_Same = tf.placeholder(dtype=tf.float32, shape=[None, same_num])
input_InDuanDis = tf.placeholder(dtype=tf.float32, shape=[None, dis_num])

input_A = tf.placeholder(dtype=tf.int32, shape=[None, action_actions])

S_all_dim = 2*edu_rnn_dim + 2*pos_rnn_dim
B_all_dim = 2*edu_rnn_dim + 2*pos_rnn_dim
all_dim = (stack_edus)*S_all_dim + (buffer_edus)*B_all_dim + action_actions*action_dim
# 整个网络结构需要用到的参数
Weights = {
    # word embedding matrix
    'word_embedding': tf.Variable(tf.constant(word_embedding_matrix), trainable=True),

    # word_Attention 参数
    'word_A_W': tf.Variable(tf.truncated_normal([2 * edu_rnn_dim], stddev=stddev_setting)),
    'word_A_b': tf.Variable(tf.truncated_normal([1], stddev=stddev_setting)),

    # pos_Attention 参数
    'pos_A_W': tf.Variable(tf.truncated_normal([2 * pos_rnn_dim], stddev=stddev_setting)),
    'pos_A_b': tf.Variable(tf.truncated_normal([1], stddev=stddev_setting)),
    # 全连接神经网络层
    'full_connect_W1': tf.Variable(tf.truncated_normal([all_dim, all_dim], stddev=stddev_setting)),
    'full_connect_b1':tf.Variable(tf.truncated_normal([all_dim], stddev=stddev_setting)),
    'full_connect_W2': tf.Variable(tf.truncated_normal([all_dim, all_dim], stddev=stddev_setting)),
    'full_connect_b2':tf.Variable(tf.truncated_normal([all_dim], stddev=stddev_setting)),
    # 最后做Softmax之前的线性变换的参数
    'W_out': tf.Variable(tf.truncated_normal([all_dim, class_Size], stddev=stddev_setting)),
    'b_out': tf.Variable(tf.truncated_normal([class_Size], stddev=stddev_setting)),

    # action embedding matrix
    'action_embedding': tf.Variable(tf.truncated_normal([action_kinds, action_dim], stddev=stddev_setting)),
    # pos embedding matrix
    'pos_embedding': tf.Variable(tf.truncated_normal([pos_kinds, pos_dim], stddev=stddev_setting)),

    # 句子号特征信息
    'Sent_num_Info': tf.Variable(tf.truncated_normal([max_position_num, position_dim], stddev=stddev_setting)),
    # 段num 特征信息
    'Duan_num_Info': tf.Variable(tf.truncated_normal([max_position_num, position_dim], stddev=stddev_setting)),
    # Text num 特征信息
    'Text_num_Info': tf.Variable(tf.truncated_normal([max_position_num, position_dim], stddev=stddev_setting)),

    # Same_list
    'Same_list_Info': tf.Variable(tf.truncated_normal([same_num, same_fea_dim], stddev=stddev_setting)),
    # In Duan Dis
    'InDuanDis_Info': tf.Variable(tf.truncated_normal([dis_num, dis_fea_dim], stddev=stddev_setting)),

    # memory parameter
    'M_S': tf.Variable(tf.truncated_normal([mem_dim, stack_edus*S_all_dim], stddev=stddev_setting)),
    'M_B': tf.Variable(tf.truncated_normal([mem_dim, buffer_edus*B_all_dim], stddev=stddev_setting))
}


# 定义网络函数
def Get_edu_embedding(input_, Scope):
    """
    获得Stack或者Buffer中edu相关的embedding信息, 每个edu过一个bi-LSTM得到一个embedding,然后多个edu连接起来
    :param input_: stack/buffer的edu信息 shape=[batch, edu_num, edu_padding_length]
    :param Scope: 指明命名域, Scope='S_edu_rnn'表明是对Stack做embedding
                             Scope='B_edu_rnn'表明是对Buffer做embedding
    :return: 返回一个representation vector shape=[batch, edu_num, 2*edu_rnn_dim]
    """
    edus_embeddings = tf.nn.embedding_lookup(Weights['word_embedding'], input_)
        # shape = [batch, edu_num, edu_padding_length, word_dim]
    edu_input_ = tf.reshape(edus_embeddings, shape=[-1, edu_padding_length, word_dim])
        # shape = [batch*edu_num, edu_padding_length, word_dim]
    edu_lstm_fw_cell = tf.contrib.rnn.LSTMCell(edu_rnn_dim)
    edu_lstm_bw_cell = tf.contrib.rnn.LSTMCell(edu_rnn_dim)
    edu_rnn_length = tf.tile([edu_padding_length], [tf.shape(input_)[0] * tf.shape(input_)[1]])
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(edu_lstm_fw_cell, edu_lstm_bw_cell, edu_input_,
                                                 sequence_length=edu_rnn_length,
                                                 dtype=tf.float32, scope=Scope)
    rnn_outs = tf.concat(outputs, 2)
    edu_W = tf.reshape(tf.tile(Weights['word_A_W'], [tf.shape(input_)[0] * tf.shape(input_)[1]]),
                       shape=[-1, 2 * edu_rnn_dim, 1])
    edu_b = Weights['word_A_b']
    edu_e = tf.transpose(tf.matmul(rnn_outs, edu_W) + edu_b, [0, 2, 1])  # [batch*edu_num, 1, edu_padding]
    word_attention = tf.nn.softmax(edu_e, -1)
    edu_embedd_result = tf.reshape(tf.matmul(word_attention, rnn_outs), [tf.shape(input_)[0], tf.shape(input_)[1], -1])
        # shape = [batch, edu_num, 2*edu_rnn_dim]
    return edu_embedd_result

def Get_pos_embedding(input_, Scope):
    """
    获得stack/buffer pos的embedding信息
    :param input_: stack/buffer的pos信息, shape=[batch, edu_num, edu_padding_length]
    :param Scope: 指明命名域, Scope='S_pos_rnn'表明是对Stack做embedding
                             Scope='B_pos_rnn'表明是对Buffer做embedding
    :return:返回一个representation vector, shape=[batch, edu_num, 2*pos_rnn_dim]
    """
    poses_embeddings = tf.nn.embedding_lookup(Weights['pos_embedding'], input_)
    pos_input_ = tf.reshape(poses_embeddings, shape=[-1, edu_padding_length, pos_dim])
    pos_lstm_fw_cell = tf.contrib.rnn.LSTMCell(pos_rnn_dim)
    pos_lstm_bw_cell = tf.contrib.rnn.LSTMCell(pos_rnn_dim)
    pos_rnn_length = tf.tile([edu_padding_length], [tf.shape(input_)[0]*tf.shape(input_)[1]])
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(pos_lstm_fw_cell, pos_lstm_bw_cell, pos_input_,
                                                 sequence_length=pos_rnn_length,
                                                 dtype=tf.float32, scope=Scope)
    rnn_outs = tf.concat(outputs, 2)
    pos_W = tf.reshape(tf.tile(Weights['pos_A_W'], [tf.shape(input_)[0] * tf.shape(input_)[1]]),
                       shape=[-1, 2 * pos_rnn_dim, 1])
    pos_b = Weights['pos_A_b']
    pos_e = tf.transpose(tf.matmul(rnn_outs, pos_W) + pos_b, [0, 2, 1])  # [batch*edu_num, 1, edu_padding]
    pos_attention = tf.nn.softmax(pos_e, -1)
    pos_embedd_result = tf.reshape(tf.matmul(pos_attention, rnn_outs), [tf.shape(input_)[0], tf.shape(input_)[1], -1])
    # shape = [batch, edu_num, 2*pos_rnn_dim]
    return pos_embedd_result

def Get_action_embedding(input_):
    """
    获得actions的embedding
    :param input_: action stack中前几个action的index shape=[batch, action_actions]
    :return: 获得每个action的embedding shape=[batch, action_actions*action_dim]
    """
    action_embeddings = tf.nn.embedding_lookup(Weights['action_embedding'], input_)
    embedding_result = tf.reshape(action_embeddings, shape=[tf.shape(input_)[0], -1])
    return embedding_result  # shape=[batch, action_actions*action_dim]

def Get_memory_embedding(input_, para_name):
    """
    从memory中获取相应的信息，主要是stack[-1]和buffer[0]在mem中的对应信息
    :param input_: stack/buffer embedding结果 shape=[batch, stack_edus/buffer_edus, S/B_all_dim]
    :param para_name: para_name='M_S' 寻找stack mem相关信息
                      para_name='M_B' 寻找buffer mem相关信息
    :return: 返回从mem中找到的结果 shape=[batch, S/B_all_dim]
    """
    input_ = tf.reshape(input_, shape=[tf.shape(input_)[0], -1])
    norm_out = tf.nn.l2_normalize(input_, 1)  # shape=[batch, stack_edus/buffer_edus*S/B_all_dim]
    Weight_norm = tf.transpose(tf.nn.l2_normalize(Weights[para_name], 1), [1, 0])  # shape=[stack_edus/buffer_edus*S/B_all_dim, mem_dim]
    mem_attention = tf.nn.softmax(mem_beta * tf.matmul(norm_out, Weight_norm), -1)  # shape=[batch, mem_dim]
    m_out = tf.matmul(mem_attention, Weights[para_name])  # shape=[batch, stack_edus/buffer_edus*S/B_all_dim]
    return m_out

def Get_embdding(input_):
    """
    对stack/buffer的edu之间的关系进行处理
    :param input_: Stack/Buffer中每个edu的info shape=[batch, stack/buffer_edus, S/B_all_dim]
    :param input_mem: Stack[-1]/Buffer[0]在memory中的信息 shape=[batch, S/B_all_dim]
    :return: Stack/Buffer最后embedding的信息
             shape=[batch, stack_edus*S_all_dim + S_all_dim] / [batch, buffer_edus*B_all_dim + B_all_dim]
    """
    Edus = tf.reshape(input_, shape=[tf.shape(input_)[0], -1])  # shape=[batch, stack/buffer_edus*S/B_all_dim]
    # embeddings = tf.concat([Edus, input_mem], 1)
        # shape=[batch, stack_edus*S_all_dim + S_all_dim] / [batch, buffer_edus*B_all_dim + B_all_dim]
    return Edus

# 获得三部分的embedding的代码
S_edu_embeddings = Get_edu_embedding(input_S, 'S_edu_rnn')  # shape=[batch, stack_edus, 2*edu_rnn_dim]
S_pos_embeddings = Get_pos_embedding(input_S_p, 'S_pos_rnn')  # shape=[batch, stack_edus, 2*pos_rnn_dim]
# S_Sent_Feat = tf.nn.embedding_lookup(Weights['Sent_num_Info'], input_S_sent)  # shape=[batch, stack_edus, position_dim]
# S_Duan_Feat = tf.nn.embedding_lookup(Weights['Duan_num_Info'], input_S_duan)  # shape=[batch, stack_edus, position_dim]
# S_Text_Feat = tf.nn.embedding_lookup(Weights['Text_num_Info'], input_S_text)  # shape=[batch, stack_edus, position_dim]
S_info = tf.concat([S_edu_embeddings, S_pos_embeddings], 2)
    # shape=[batch, stack_edus, S_all_dim] S_all_dim = 2*edu_rnn_dim + 2*pos_rnn_dim + 3*position_dim

B_edu_embeddings = Get_edu_embedding(input_B, 'B_edu_rnn')
B_pos_embeddings = Get_pos_embedding(input_B_p, 'B_pos_rnn')
# B_Sent_Feat = tf.nn.embedding_lookup(Weights['Sent_num_Info'], input_B_sent)  # shape=[batch, buffer_edus, position_dim]
# B_Duan_Feat = tf.nn.embedding_lookup(Weights['Duan_num_Info'], input_B_duan)  # shape=[batch, buffer_edus, position_dim]
# B_Text_Feat = tf.nn.embedding_lookup(Weights['Text_num_Info'], input_B_text)  # shape=[batch, buffer_edus, position_dim]
B_info = tf.concat([B_edu_embeddings, B_pos_embeddings], 2)
    # shape=[batch, buffer_edus, B_all_dim] B_all_dim = 2*edu_rnn_dim + 2*pos_rnn_dim + 3*position_dim

# Same_Feat = tf.matmul(input_Same, Weights['Same_list_Info'])  # shape=[batch, same_fea_dim]
# Induan_Feat = tf.matmul(input_InDuanDis, Weights['InDuanDis_Info'])  # shape=[batch, dis_fea_dim]

# 获得Stack[-1], Buffer[0]在memory中的信息
# S_mem_info = Get_memory_embedding(S_info, 'M_S')
# B_mem_info = Get_memory_embedding(B_info, 'M_B')
# 将向量链接起来
S_embeddings = Get_embdding(S_info)  # shape=[batch, stack_edus*S_all_dim + S_all_dim]
B_embeddings = Get_embdding(B_info)  # shape=[batch, buffer_edus*B_all_dim + B_all_dim]
A_embeddings = Get_action_embedding(input_A)  # shape=[batch, action_actions*action_dim]
embedding_concat = tf.concat([S_embeddings, B_embeddings, A_embeddings], 1)
    # shape=[(stack_edus+1)*S_all_dim + (buffer_edus+1)*B_all_dim + action_actions*action_dim + same_feat_dim + dis_fea_dim]
# 激活函数层
embedding_concat = tf.nn.relu(embedding_concat, name='embedding_relu')
# 全连接线性网络层
full_connect_layer1 = tf.matmul(embedding_concat, Weights['full_connect_W1'])+Weights['full_connect_b1']
# 激活函数层
full_connect_layer1 = tf.nn.relu(full_connect_layer1)
# 全连接线性网络层
full_connect_layer2 = tf.matmul(full_connect_layer1, Weights['full_connect_W2'])+Weights['full_connect_b2']
# 激活函数层
full_connect_layer2 = tf.nn.relu(full_connect_layer2)
# 线性变换
hidden_layer_out = tf.matmul(full_connect_layer2, Weights['W_out']) + Weights['b_out']
l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
# 计算cost，进行梯度下降训练
gold_Ac = tf.placeholder(tf.float32, shape=[None, class_Size])
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gold_Ac, logits=hidden_layer_out))
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(lr, global_step, 5000, 0.96, staircase=True)
# Passing global_step to minimize() will increment it at each step.
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

# 保存模型相关，tf的运行相关的设置
saver = tf.train.Saver(max_to_keep=1)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


# train操作
def train_single(S_v_lists, S_p_lists, S_s_num, S_d_num, S_t_num,
                 B_v_lists, B_p_lists, B_s_num, B_d_num, B_t_num,
                 same_l, induandis, A_v_lists):
    predict = sess.run(hidden_layer_out, feed_dict={
        input_S: S_v_lists, input_S_p: S_p_lists,
        input_S_sent: S_s_num, input_S_duan: S_d_num, input_S_text: S_t_num,
        input_B: B_v_lists, input_B_p: B_p_lists,
        input_B_sent: B_s_num, input_B_duan: B_d_num, input_B_text: B_t_num,
        input_Same: same_l, input_InDuanDis: induandis,
        input_A: A_v_lists})
    return predict


def train_update(S_v_lists, S_p_lists, S_s_num, S_d_num, S_t_num,
                 B_v_lists, B_p_lists, B_s_num, B_d_num, B_t_num,
                 same_l, induandis, A_v_lists, gold_actions):
    global  Globla_Step
    Globla_Step += 1
    predict, _, b_out, t_loss, t_l2_loss, LR = sess.run([hidden_layer_out, train_step, Weights['b_out'], cost, l2_losses, learning_rate], feed_dict={
        input_S: S_v_lists, input_S_p: S_p_lists,
        input_S_sent: S_s_num, input_S_duan: S_d_num, input_S_text: S_t_num,
        input_B: B_v_lists, input_B_p: B_p_lists,
        input_B_sent: B_s_num, input_B_duan: B_d_num, input_B_text: B_t_num,
        input_Same: same_l, input_InDuanDis: induandis,
        input_A: A_v_lists, gold_Ac: gold_actions})
    return predict

def train_model_save(Iter):
    save_path = "Model/" + "Iter" + str(Iter) + "/model.ckpt"
    if not os.path.isdir("Model/" + "Iter" + str(Iter)):
        os.mkdir("Model/" + "Iter" + str(Iter))
    saver.save(sess, save_path)
    print ("Model stored....")


def train_model_import(Iter):
    save_path = "Model/" + "Iter" + str(Iter) + "/model.ckpt"
    if not os.path.isdir("Model/" + "Iter" + str(Iter)):
        os.mkdir("Model/" + "Iter" + str(Iter))
    saver.restore(sess, save_path)
    print("Model restored.")
