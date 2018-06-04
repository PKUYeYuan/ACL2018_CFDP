# coding=utf-8
import tensorflow as tf
from yy_flags import  Flags
# define the neural network model of our parser
class Basic_model:
    def __init__(self, is_training, word_embedding):
        self.input_S_word = tf.placeholder(dtype=tf.int32, shape=[None, Flags.Stack_edus, Flags.edu_padding_length],
                                           name='input_S_word')
        self.input_S_POS = tf.placeholder(dtype=tf.int32, shape=[None, Flags.Stack_edus, Flags.edu_padding_length],
                                          name='input_S_POS')
        self.input_B_word = tf.placeholder(dtype=tf.int32, shape=[None, Flags.Buffer_edus, Flags.edu_padding_length],
                                           name='input_B_word')
        self.input_B_POS = tf.placeholder(dtype=tf.int32, shape=[None, Flags.Buffer_edus, Flags.edu_padding_length],
                                           name='input_B_POS')
        self.input_A_his = tf.placeholder(dtype=tf.int32, shape=[None, Flags.Action_actions],
                                          name='input_A_his')
        if is_training:
            self.gold_action = tf.placeholder(dtype=tf.int32, shape=[None, Flags.Class_size], name='gold_action')
            self.word_embedding = tf.get_variable('word_embedding', initializer=word_embedding, dtype=tf.float32)
        else:
            self.word_embedding = tf.get_variable('word_embedding', shape=[Flags.vocab_size, Flags.word_vec_dim],
                                                  dtype=tf.float32)
        # get Stack vector representation
        with tf.variable_scope("S_repre"):
            self.S_word_repre = self.Get_repre_word(self.input_S_word, 'S_word_rnn')
            self.S_POS_repre = self.Get_repre_POS(self.input_S_POS, 'S_POS_rnn')
            self.S_repre = tf.reshape(tf.concat([self.S_word_repre, self.S_POS_repre], 2),
                                      shape=[tf.shape(self.input_S_word)[0], -1])
        # get Buffer vector representation
        with tf.variable_scope("B_repre"):
            self.B_word_repre = self.Get_repre_word(self.input_B_word, 'B_word_rnn')
            self.B_POS_repre = self.Get_repre_POS(self.input_B_POS, 'B_POS_rnn')
            self.B_repre = tf.reshape(tf.concat([self.B_word_repre, self.B_POS_repre], 2),
                                      shape=[tf.shape(self.input_B_word)[0], -1])
        # get Action History Stack vector representation
        self.A_repre = self.Get_action_repre(self.input_A_his)
        # concat the three part to get a configure representation
        self.concat_conf_repre = tf.nn.relu(tf.concat([self.S_repre, self.B_repre, self.A_repre], 1))
        # pass configure repre into MLP(two full connect layer with relu)
        self.MLP_conf_repre = self.MLP_layer(self.concat_conf_repre)
        self.out_repre = self.get_out(self.MLP_conf_repre)
        self.prob = self.get_prob(self.out_repre)
        self.prediction = self.get_prediction(self.prob)
        if is_training:
            self.accuracy = self.get_accuracy(self.prediction)
            self.en_loss = self.get_entropy_loss(self.out_repre)

    def Get_repre_word(self, input_, Scope):
        """
        get the vector representation of Stack(or Buffer) based on the word information
        first we get edu embedding by bi-lstm with attention mechanism
        then we concat each edu representation to get Stack(or Buffer) representation
        """
        word_embedding_info = tf.nn.embedding_lookup(self.word_embedding, input_)
        word_input_ = tf.reshape(word_embedding_info, shape=[-1, Flags.edu_padding_length, Flags.word_vec_dim])
                                                   # shape = [batch*edu_num, edu_padding_length, word_dim]
        word_lstm_fw_cell = tf.contrib.rnn.LSTMCell(Flags.word_rnn_dim)
        word_lstm_bw_cell = tf.contrib.rnn.LSTMCell(Flags.word_rnn_dim)
        word_rnn_length = tf.tile([Flags.edu_padding_length], [tf.shape(input_)[0] * tf.shape(input_)[1]])
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(word_lstm_fw_cell, word_lstm_bw_cell, word_input_,
                                                     sequence_length=word_rnn_length,
                                                     dtype=tf.float32, scope=Scope)
        rnn_outs = tf.concat(outputs, 2)

        # EDU embedding word attention parameters
        word_bilstm_A_W = tf.get_variable('word_bilstm_A_W', shape=[2 * Flags.word_rnn_dim])
        word_bilstm_A_b = tf.get_variable('word_bilstm_A_b', shape=[1])

        word_W = tf.reshape(tf.tile(word_bilstm_A_W, [tf.shape(input_)[0] * tf.shape(input_)[1]]),
                            shape=[-1, 2 * Flags.word_rnn_dim, 1])
        word_b = word_bilstm_A_b
        word_e = tf.transpose(tf.matmul(rnn_outs, word_W) + word_b, [0, 2, 1])  # [batch*edu_num, 1, edu_padding]
        word_attention = tf.nn.softmax(word_e, -1)
        edu_embedd_result = tf.reshape(tf.matmul(word_attention, rnn_outs),
                                       [tf.shape(input_)[0], tf.shape(input_)[1], -1])
        return edu_embedd_result

    def Get_repre_POS(self, input_, Scope):
        """
        get the vector representation of Stack(or Buffer) based on the POS information
        first we get edu embedding by bi-lstm with attention mechanism
        then we concat each edu representation to get Stack(or Buffer) representation
        """
        POS_embedding = tf.get_variable('POS_emebdding', shape=[Flags.POS_kinds, Flags.POS_embedding_dim])
        POS_embedding_info = tf.nn.embedding_lookup(POS_embedding, input_)
        POS_input_ = tf.reshape(POS_embedding_info, shape=[-1, Flags.edu_padding_length, Flags.POS_embedding_dim])
        POS_lstm_fw_cell = tf.contrib.rnn.LSTMCell(Flags.POS_rnn_dim)
        POS_lstm_bw_cell = tf.contrib.rnn.LSTMCell(Flags.POS_rnn_dim)
        POS_rnn_length = tf.tile([Flags.edu_padding_length], [tf.shape(input_)[0] * tf.shape(input_)[1]])
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(POS_lstm_fw_cell, POS_lstm_bw_cell, POS_input_,
                                                     sequence_length=POS_rnn_length,
                                                     dtype=tf.float32, scope=Scope)
        rnn_outs = tf.concat(outputs, 2)

        # EDU embedding POS attention parameters
        POS_bilstm_A_W = tf.get_variable('POS_bilstm_A_W', shape=[2 * Flags.POS_rnn_dim])
        POS_bilstm_A_b = tf.get_variable('POS_bilstm_A_b', shape=[1])
        POS_W = tf.reshape(tf.tile(POS_bilstm_A_W, [tf.shape(input_)[0] * tf.shape(input_)[1]]),
                           shape=[-1, 2 * Flags.POS_rnn_dim, 1])
        POS_b = POS_bilstm_A_b
        POS_e = tf.transpose(tf.matmul(rnn_outs, POS_W) + POS_b, [0, 2, 1])  # [batch*edu_num, 1, edu_padding]
        POS_attention = tf.nn.softmax(POS_e, -1)
        POS_embedd_result = tf.reshape(tf.matmul(POS_attention, rnn_outs),
                                       [tf.shape(input_)[0], tf.shape(input_)[1], -1])
        return POS_embedd_result

    def Get_action_repre(self, input_):
        """
        get the vector representation of Action History Stack
        we just simple concat each action vector representation
        """
        action_embedding = tf.get_variable('action_embedding', shape=[Flags.action_history_kinds+1, Flags.action_embedding_dim])
        action_embedding_info = tf.nn.embedding_lookup(action_embedding, input_)
        embedding_result = tf.reshape(action_embedding_info, shape=[tf.shape(input_)[0], -1])
        return embedding_result

    def MLP_layer(self, input_):
        '''
        pass the configure_repre into two full connect layer with relu as their activation function
        '''
        # full connect layer parameters
        all_dim = Flags.Stack_edus*(2*Flags.word_rnn_dim+2*Flags.POS_rnn_dim) + \
                  Flags.Buffer_edus*(2*Flags.word_rnn_dim+2*Flags.POS_rnn_dim) + \
                  Flags.Action_actions*(Flags.action_embedding_dim)
        FC_W1 = tf.get_variable('Full_Connect_layer_W1', shape=[all_dim, all_dim])
        FC_b1 = tf.get_variable('Full_Connect_layer_b1', shape=[all_dim])
        FC_W2 = tf.get_variable('Full_Connect_layer_W2', shape=[all_dim, all_dim])
        FC_b2 = tf.get_variable('Full_Connect_layer_b2', shape=[all_dim])
        FC1 = tf.nn.relu(tf.matmul(input_, FC_W1)+FC_b1)
        FC2 = tf.nn.relu(tf.matmul(FC1, FC_W2)+FC_b2)
        return FC2

    def get_out(self, input_):
        all_dim = Flags.Stack_edus * (2 * Flags.word_rnn_dim + 2 * Flags.POS_rnn_dim) + \
                  Flags.Buffer_edus * (2 * Flags.word_rnn_dim + 2 * Flags.POS_rnn_dim) + \
                  Flags.Action_actions * (Flags.action_embedding_dim)
        W_out = tf.get_variable('W_out', shape=[all_dim, Flags.Class_size])
        b_out = tf.get_variable('b_out', shape=[Flags.Class_size])
        out = tf.matmul(input_, W_out) + b_out
        return out

    def get_prob(self, input_):
        prob = tf.nn.softmax(input_, -1)
        return prob

    def get_prediction(self, input_):
        predictios = tf.argmax(input_, 1)
        return predictios

    def get_accuracy(self, input_):
        accuracy = tf.cast(tf.equal(self.prediction, tf.argmax(self.gold_action, 1)), "float")
        return accuracy

    def get_entropy_loss(self, input_):
        entropy_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=input_, labels=self.gold_action))
        return entropy_loss

class Basic_model_ex:
    def __init__(self, is_training, word_embedding):
        self.position1dim = 4
        self.position1_embedding_dim = 15
        self.same_vector_dim = 4
        self.same_embedding_dim = 15
        self.para_dis_dim = 17
        self.para_dis_embedding_dim = 15
        self.input_S_word = tf.placeholder(dtype=tf.int32, shape=[None, Flags.Stack_edus, Flags.edu_padding_length],
                                           name='input_S_word')
        self.input_S_POS = tf.placeholder(dtype=tf.int32, shape=[None, Flags.Stack_edus, Flags.edu_padding_length],
                                          name='input_S_POS')
        self.input_S_sent = tf.placeholder(dtype=tf.int32, shape=[None, Flags.Stack_edus], name='input_S_sent')
        self.input_S_para = tf.placeholder(dtype=tf.int32, shape=[None, Flags.Stack_edus], name='input_S_para')
        self.input_S_disc = tf.placeholder(dtype=tf.int32, shape=[None, Flags.Stack_edus], name='input_S_disc')

        self.input_B_word = tf.placeholder(dtype=tf.int32, shape=[None, Flags.Buffer_edus, Flags.edu_padding_length],
                                           name='input_B_word')
        self.input_B_POS = tf.placeholder(dtype=tf.int32, shape=[None, Flags.Buffer_edus, Flags.edu_padding_length],
                                           name='input_B_POS')
        self.input_B_sent = tf.placeholder(dtype=tf.int32, shape=[None, Flags.Buffer_edus], name='input_B_sent')
        self.input_B_para = tf.placeholder(dtype=tf.int32, shape=[None, Flags.Buffer_edus], name='input_B_para')
        self.input_B_disc = tf.placeholder(dtype=tf.int32, shape=[None, Flags.Buffer_edus], name='input_B_disc')

        self.input_A_his = tf.placeholder(dtype=tf.int32, shape=[None, Flags.Action_actions],
                                          name='input_A_his')
        self.input_same = tf.placeholder(dtype=tf.float32, shape=[None, self.same_vector_dim], name='input_same')
        self.input_para_dis = tf.placeholder(dtype=tf.float32, shape=[None, self.para_dis_dim], name='input_para_dis')

        if is_training:
            self.gold_action = tf.placeholder(dtype=tf.int32, shape=[None, Flags.Class_size], name='gold_action')
            self.word_embedding = tf.get_variable('word_embedding', initializer=word_embedding, dtype=tf.float32)
        else:
            self.word_embedding = tf.get_variable('word_embedding', shape=[Flags.vocab_size, Flags.word_vec_dim],
                                                  dtype=tf.float32)
        self.POS_embedding = tf.get_variable('POS_emebdding', shape=[Flags.POS_kinds, Flags.POS_embedding_dim])
        self.sent_embedding = tf.get_variable('sent_embedding', shape=[self.position1dim, self.position1_embedding_dim])
        self.para_embedding = tf.get_variable('para_embedding', shape=[self.position1dim, self.position1_embedding_dim])
        self.disc_embedding = tf.get_variable('disc_embedding', shape=[self.position1dim, self.position1_embedding_dim])
        self.same_embedding = tf.get_variable('same_embedding', shape=[self.same_vector_dim, self.same_embedding_dim])
        self.para_dis_embedding = tf.get_variable('para_dis_embedding',
                                                  shape=[self.para_dis_dim, self.para_dis_embedding_dim])
        # get Stack vector representation
        self.S_word_repre = self.Get_repre_word(self.input_S_word, 'S_word_rnn')
        self.S_POS_repre = self.Get_repre_POS(self.input_S_POS, 'S_POS_rnn')
        self.S_position_fea1_repre = self.Get_position_fea1_repre(self.input_S_sent, self.input_S_para,
                                                                  self.input_S_disc)
        self.S_repre = tf.reshape(tf.concat([self.S_word_repre, self.S_POS_repre, self.S_position_fea1_repre], 2),
                                  shape=[tf.shape(self.input_S_word)[0], -1])

        # get Buffer vector representation
        self.B_word_repre = self.Get_repre_word(self.input_B_word, 'B_word_rnn')
        self.B_POS_repre = self.Get_repre_POS(self.input_B_POS, 'B_POS_rnn')
        self.B_position_fea1_repre = self.Get_position_fea1_repre(self.input_B_sent, self.input_B_para,
                                                                  self.input_B_disc)
        self.B_repre = tf.reshape(tf.concat([self.B_word_repre, self.B_POS_repre, self.B_position_fea1_repre], 2),
                                  shape=[tf.shape(self.input_B_word)[0], -1])

        # get Action History Stack vector representation
        self.A_repre = self.Get_action_repre(self.input_A_his)
        self.same_fea = tf.matmul(self.input_same, self.same_embedding)
        self.para_dis_fea = tf.matmul(self.input_para_dis, self.para_dis_embedding)

        # concat the three part to get a configure representation
        self.concat_conf_repre = tf.nn.relu(tf.concat(
            [self.S_repre, self.B_repre, self.A_repre, self.same_fea, self.para_dis_fea], 1))
        # pass configure repre into MLP(two full connect layer with relu)
        self.MLP_conf_repre = self.MLP_layer(self.concat_conf_repre)
        self.out_repre = self.get_out(self.MLP_conf_repre)
        self.prob = self.get_prob(self.out_repre)
        self.prediction = self.get_prediction(self.prob)
        if is_training:
            self.accuracy = self.get_accuracy(self.prediction)
            self.en_loss = self.get_entropy_loss(self.out_repre)

    def Get_repre_word(self, input_, Scope):
        """
        get the vector representation of Stack(or Buffer) based on the word information
        first we get edu embedding by bi-lstm with attention mechanism
        then we concat each edu representation to get Stack(or Buffer) representation
        """
        word_embedding_info = tf.nn.embedding_lookup(self.word_embedding, input_)
        word_input_ = tf.reshape(word_embedding_info, shape=[-1, Flags.edu_padding_length, Flags.word_vec_dim])
                                                   # shape = [batch*edu_num, edu_padding_length, word_dim]
        word_lstm_fw_cell = tf.contrib.rnn.LSTMCell(Flags.word_rnn_dim)
        word_lstm_bw_cell = tf.contrib.rnn.LSTMCell(Flags.word_rnn_dim)
        word_rnn_length = tf.tile([Flags.edu_padding_length], [tf.shape(input_)[0] * tf.shape(input_)[1]])
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(word_lstm_fw_cell, word_lstm_bw_cell, word_input_,
                                                     sequence_length=word_rnn_length,
                                                     dtype=tf.float32, scope=Scope)
        rnn_outs = tf.concat(outputs, 2)

        # EDU embedding word attention parameters
        word_bilstm_A_W = tf.get_variable(Scope+'_'+'word_bilstm_A_W', shape=[2 * Flags.word_rnn_dim])
        word_bilstm_A_b = tf.get_variable(Scope+'_'+'word_bilstm_A_b', shape=[1])

        word_W = tf.reshape(tf.tile(word_bilstm_A_W, [tf.shape(input_)[0] * tf.shape(input_)[1]]),
                            shape=[-1, 2 * Flags.word_rnn_dim, 1])
        word_b = word_bilstm_A_b
        word_e = tf.transpose(tf.matmul(rnn_outs, word_W) + word_b, [0, 2, 1])  # [batch*edu_num, 1, edu_padding]
        word_attention = tf.nn.softmax(word_e, -1)
        edu_embedd_result = tf.reshape(tf.matmul(word_attention, rnn_outs),
                                       [tf.shape(input_)[0], tf.shape(input_)[1], -1])
        return edu_embedd_result

    def Get_repre_POS(self, input_, Scope):
        """
        get the vector representation of Stack(or Buffer) based on the POS information
        first we get edu embedding by bi-lstm with attention mechanism
        then we concat each edu representation to get Stack(or Buffer) representation
        """
        POS_embedding_info = tf.nn.embedding_lookup(self.POS_embedding, input_)
        POS_input_ = tf.reshape(POS_embedding_info, shape=[-1, Flags.edu_padding_length, Flags.POS_embedding_dim])
        POS_lstm_fw_cell = tf.contrib.rnn.LSTMCell(Flags.POS_rnn_dim)
        POS_lstm_bw_cell = tf.contrib.rnn.LSTMCell(Flags.POS_rnn_dim)
        POS_rnn_length = tf.tile([Flags.edu_padding_length], [tf.shape(input_)[0] * tf.shape(input_)[1]])
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(POS_lstm_fw_cell, POS_lstm_bw_cell, POS_input_,
                                                     sequence_length=POS_rnn_length,
                                                     dtype=tf.float32, scope=Scope)
        rnn_outs = tf.concat(outputs, 2)

        # EDU embedding POS attention parameters
        POS_bilstm_A_W = tf.get_variable(Scope+'_'+'POS_bilstm_A_W', shape=[2 * Flags.POS_rnn_dim])
        POS_bilstm_A_b = tf.get_variable(Scope+'_'+'POS_bilstm_A_b', shape=[1])
        POS_W = tf.reshape(tf.tile(POS_bilstm_A_W, [tf.shape(input_)[0] * tf.shape(input_)[1]]),
                           shape=[-1, 2 * Flags.POS_rnn_dim, 1])
        POS_b = POS_bilstm_A_b
        POS_e = tf.transpose(tf.matmul(rnn_outs, POS_W) + POS_b, [0, 2, 1])  # [batch*edu_num, 1, edu_padding]
        POS_attention = tf.nn.softmax(POS_e, -1)
        POS_embedd_result = tf.reshape(tf.matmul(POS_attention, rnn_outs),
                                       [tf.shape(input_)[0], tf.shape(input_)[1], -1])
        return POS_embedd_result

    def Get_action_repre(self, input_):
        """
        get the vector representation of Action History Stack
        we just simple concat each action vector representation
        """
        action_embedding = tf.get_variable('action_embedding', shape=[Flags.Class_size+1, Flags.action_embedding_dim])
        action_embedding_info = tf.nn.embedding_lookup(action_embedding, input_)
        embedding_result = tf.reshape(action_embedding_info, shape=[tf.shape(input_)[0], -1])
        return embedding_result

    def Get_position_fea1_repre(self, sent, para, disc):
        sent_repre = tf.nn.embedding_lookup(self.sent_embedding, sent)
        para_repre = tf.nn.embedding_lookup(self.para_embedding, para)
        disc_repre = tf.nn.embedding_lookup(self.disc_embedding, disc)
        return tf.concat([sent_repre, para_repre, disc_repre], 2)

    def MLP_layer(self, input_):
        '''
        pass the configure_repre into two full connect layer with relu as their activation function
        '''
        # full connect layer parameters
        all_dim = Flags.Stack_edus*(2*Flags.word_rnn_dim+2*Flags.POS_rnn_dim+3*self.position1_embedding_dim) + \
                  Flags.Buffer_edus*(2*Flags.word_rnn_dim+2*Flags.POS_rnn_dim+3*self.position1_embedding_dim) + \
                  Flags.Action_actions*(Flags.action_embedding_dim)+ \
                  self.same_embedding_dim + self.para_dis_embedding_dim
        FC_W1 = tf.get_variable('Full_Connect_layer_W1', shape=[all_dim, all_dim])
        FC_b1 = tf.get_variable('Full_Connect_layer_b1', shape=[all_dim])
        FC_W2 = tf.get_variable('Full_Connect_layer_W2', shape=[all_dim, all_dim])
        FC_b2 = tf.get_variable('Full_Connect_layer_b2', shape=[all_dim])
        FC1 = tf.nn.relu(tf.matmul(input_, FC_W1)+FC_b1)
        FC2 = tf.nn.relu(tf.matmul(FC1, FC_W2)+FC_b2)
        return FC2

    def get_out(self, input_):
        all_dim = Flags.Stack_edus * (2 * Flags.word_rnn_dim + 2 * Flags.POS_rnn_dim+3*self.position1_embedding_dim) + \
                  Flags.Buffer_edus * (2 * Flags.word_rnn_dim + 2 * Flags.POS_rnn_dim+3*self.position1_embedding_dim) + \
                  Flags.Action_actions * (Flags.action_embedding_dim)+ \
                  self.same_embedding_dim + self.para_dis_embedding_dim
        W_out = tf.get_variable('W_out', shape=[all_dim, Flags.Class_size])
        b_out = tf.get_variable('b_out', shape=[Flags.Class_size])
        out = tf.matmul(input_, W_out) + b_out
        return out

    def get_prob(self, input_):
        prob = tf.nn.softmax(input_, -1)
        return prob

    def get_prediction(self, input_):
        predictios = tf.argmax(input_, 1)
        return predictios

    def get_accuracy(self, input_):
        accuracy = tf.cast(tf.equal(self.prediction, tf.argmax(self.gold_action, 1)), "float")
        return accuracy

    def get_entropy_loss(self, input_):
        entropy_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=input_, labels=self.gold_action))
        return entropy_loss

class Refine_mem_model:
    def __init__(self, is_training, word_embedding):
        self.position1dim = 4
        self.position1_embedding_dim = 15
        self.same_vector_dim = 4
        self.same_embedding_dim = 15
        self.para_dis_dim = 17
        self.para_dis_embedding_dim = 15
        self.mem_slots = 20
        self.mem_beta = 15

        self.input_S_word = tf.placeholder(dtype=tf.int32, shape=[None, Flags.Stack_edus, Flags.edu_padding_length],
                                           name='input_S_word')
        self.input_S_POS = tf.placeholder(dtype=tf.int32, shape=[None, Flags.Stack_edus, Flags.edu_padding_length],
                                          name='input_S_POS')
        self.input_S_sent = tf.placeholder(dtype=tf.int32, shape=[None, Flags.Stack_edus], name='input_S_sent')
        self.input_S_para = tf.placeholder(dtype=tf.int32, shape=[None, Flags.Stack_edus], name='input_S_para')
        self.input_S_disc = tf.placeholder(dtype=tf.int32, shape=[None, Flags.Stack_edus], name='input_S_disc')

        self.input_B_word = tf.placeholder(dtype=tf.int32, shape=[None, Flags.Buffer_edus, Flags.edu_padding_length],
                                           name='input_B_word')
        self.input_B_POS = tf.placeholder(dtype=tf.int32, shape=[None, Flags.Buffer_edus, Flags.edu_padding_length],
                                           name='input_B_POS')
        self.input_B_sent = tf.placeholder(dtype=tf.int32, shape=[None, Flags.Buffer_edus], name='input_B_sent')
        self.input_B_para = tf.placeholder(dtype=tf.int32, shape=[None, Flags.Buffer_edus], name='input_B_para')
        self.input_B_disc = tf.placeholder(dtype=tf.int32, shape=[None, Flags.Buffer_edus], name='input_B_disc')

        self.input_A_his = tf.placeholder(dtype=tf.int32, shape=[None, Flags.Action_actions],
                                          name='input_A_his')
        self.input_same = tf.placeholder(dtype=tf.float32, shape=[None, self.same_vector_dim], name='input_same')
        self.input_para_dis = tf.placeholder(dtype=tf.float32, shape=[None, self.para_dis_dim], name='input_para_dis')

        if is_training:
            self.gold_action = tf.placeholder(dtype=tf.int32, shape=[None, Flags.Class_size], name='gold_action')
            self.word_embedding = tf.get_variable('word_embedding', initializer=word_embedding, dtype=tf.float32)
        else:
            self.word_embedding = tf.get_variable('word_embedding', shape=[Flags.vocab_size, Flags.word_vec_dim],
                                                  dtype=tf.float32)
        self.POS_embedding = tf.get_variable('POS_emebdding', shape=[Flags.POS_kinds, Flags.POS_embedding_dim])
        self.sent_embedding = tf.get_variable('sent_embedding', shape=[self.position1dim, self.position1_embedding_dim])
        self.para_embedding = tf.get_variable('para_embedding', shape=[self.position1dim, self.position1_embedding_dim])
        self.disc_embedding = tf.get_variable('disc_embedding', shape=[self.position1dim, self.position1_embedding_dim])
        self.same_embedding = tf.get_variable('same_embedding', shape=[self.same_vector_dim, self.same_embedding_dim])
        self.para_dis_embedding = tf.get_variable('para_dis_embedding', shape=[self.para_dis_dim, self.para_dis_embedding_dim])

        # get Stack vector representation
        self.S_repre = self.Get_S_repre()
        # get Buffer vector representation
        self.B_repre = self.Get_B_repre()
        # get Action History Stack vector representation
        self.A_repre = self.Get_action_repre(self.input_A_his)
        self.same_fea = tf.matmul(self.input_same, self.same_embedding)
        self.para_dis_fea = tf.matmul(self.input_para_dis, self.para_dis_embedding)

        # concat the three part to get a configure representation
        self.concat_conf_repre = tf.nn.relu(tf.concat(
            [self.S_repre, self.B_repre, self.A_repre, self.same_fea, self.para_dis_fea], 1))
        # pass configure repre into MLP(two full connect layer with relu)
        self.MLP_conf_repre = self.MLP_layer(self.concat_conf_repre)
        self.out_repre = self.get_out(self.MLP_conf_repre)
        self.prob = self.get_prob(self.out_repre)
        self.prediction = self.get_prediction(self.prob)
        if is_training:
            self.accuracy = self.get_accuracy(self.prediction)
            self.en_loss = self.get_entropy_loss(self.out_repre)

    def Get_repre_word(self, input_, Scope):
        """
        get the vector representation of Stack(or Buffer) based on the word information
        first we get edu embedding by bi-lstm with attention mechanism
        then we concat each edu representation to get Stack(or Buffer) representation
        """
        word_embedding_info = tf.nn.embedding_lookup(self.word_embedding, input_)
        word_input_ = tf.reshape(word_embedding_info, shape=[-1, Flags.edu_padding_length, Flags.word_vec_dim])
                                                   # shape = [batch*edu_num, edu_padding_length, word_dim]
        word_lstm_fw_cell = tf.contrib.rnn.LSTMCell(Flags.word_rnn_dim)
        word_lstm_bw_cell = tf.contrib.rnn.LSTMCell(Flags.word_rnn_dim)
        word_rnn_length = tf.tile([Flags.edu_padding_length], [tf.shape(input_)[0] * tf.shape(input_)[1]])
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(word_lstm_fw_cell, word_lstm_bw_cell, word_input_,
                                                     sequence_length=word_rnn_length,
                                                     dtype=tf.float32, scope=Scope)
        rnn_outs = tf.concat(outputs, 2)

        # EDU embedding word attention parameters
        word_bilstm_A_W = tf.get_variable(Scope+'_'+'word_bilstm_A_W', shape=[2 * Flags.word_rnn_dim])
        word_bilstm_A_b = tf.get_variable(Scope+'_'+'word_bilstm_A_b', shape=[1])

        word_W = tf.reshape(tf.tile(word_bilstm_A_W, [tf.shape(input_)[0] * tf.shape(input_)[1]]),
                            shape=[-1, 2 * Flags.word_rnn_dim, 1])
        word_b = word_bilstm_A_b
        word_e = tf.transpose(tf.matmul(rnn_outs, word_W) + word_b, [0, 2, 1])  # [batch*edu_num, 1, edu_padding]
        word_attention = tf.nn.softmax(word_e, -1)
        edu_embedd_result = tf.reshape(tf.matmul(word_attention, rnn_outs),
                                       [tf.shape(input_)[0], tf.shape(input_)[1], -1])
        return edu_embedd_result

    def Get_repre_POS(self, input_, Scope):
        """
        get the vector representation of Stack(or Buffer) based on the POS information
        first we get edu embedding by bi-lstm with attention mechanism
        then we concat each edu representation to get Stack(or Buffer) representation
        """
        POS_embedding_info = tf.nn.embedding_lookup(self.POS_embedding, input_)
        POS_input_ = tf.reshape(POS_embedding_info, shape=[-1, Flags.edu_padding_length, Flags.POS_embedding_dim])
        POS_lstm_fw_cell = tf.contrib.rnn.LSTMCell(Flags.POS_rnn_dim)
        POS_lstm_bw_cell = tf.contrib.rnn.LSTMCell(Flags.POS_rnn_dim)
        POS_rnn_length = tf.tile([Flags.edu_padding_length], [tf.shape(input_)[0] * tf.shape(input_)[1]])
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(POS_lstm_fw_cell, POS_lstm_bw_cell, POS_input_,
                                                     sequence_length=POS_rnn_length,
                                                     dtype=tf.float32, scope=Scope)
        rnn_outs = tf.concat(outputs, 2)

        # EDU embedding POS attention parameters
        POS_bilstm_A_W = tf.get_variable(Scope+'_'+'POS_bilstm_A_W', shape=[2 * Flags.POS_rnn_dim])
        POS_bilstm_A_b = tf.get_variable(Scope+'_'+'POS_bilstm_A_b', shape=[1])
        POS_W = tf.reshape(tf.tile(POS_bilstm_A_W, [tf.shape(input_)[0] * tf.shape(input_)[1]]),
                           shape=[-1, 2 * Flags.POS_rnn_dim, 1])
        POS_b = POS_bilstm_A_b
        POS_e = tf.transpose(tf.matmul(rnn_outs, POS_W) + POS_b, [0, 2, 1])  # [batch*edu_num, 1, edu_padding]
        POS_attention = tf.nn.softmax(POS_e, -1)
        POS_embedd_result = tf.reshape(tf.matmul(POS_attention, rnn_outs),
                                       [tf.shape(input_)[0], tf.shape(input_)[1], -1])
        return POS_embedd_result

    def Get_S_mem_embedding(self, input_):
        S_all_dim = Flags.Stack_edus*(2*Flags.word_rnn_dim+2*Flags.POS_rnn_dim+3*self.position1_embedding_dim)
        S_mem_metrics=tf.get_variable('S_mem_metrics', shape=[self.mem_slots, S_all_dim])
        input_norm = tf.nn.l2_normalize(input_, 1)
        mem_norm = tf.transpose(tf.nn.l2_normalize(S_mem_metrics, 1), [1,0])
        mem_attention = tf.nn.softmax(self.mem_beta * tf.matmul(input_norm, mem_norm), -1)
        S_coh = tf.matmul(mem_attention, S_mem_metrics)
        return S_coh

    def Get_B_mem_embedding(self, input_):
        B_all_dim = Flags.Buffer_edus*(2*Flags.word_rnn_dim+2*Flags.POS_rnn_dim+3*self.position1_embedding_dim)
        B_mem_metrics=tf.get_variable('B_mem_metrics', shape=[self.mem_slots, B_all_dim])
        input_norm = tf.nn.l2_normalize(input_, 1)
        mem_norm = tf.transpose(tf.nn.l2_normalize(B_mem_metrics, 1), [1,0])
        mem_attention = tf.nn.softmax(self.mem_beta * tf.matmul(input_norm, mem_norm), -1)
        B_coh = tf.matmul(mem_attention, B_mem_metrics)
        return B_coh

    def Get_S_repre(self):
        # get Stack vector representation
        self.S_word_repre = self.Get_repre_word(self.input_S_word, 'S_word_rnn')
        self.S_POS_repre = self.Get_repre_POS(self.input_S_POS, 'S_POS_rnn')
        self.S_position_fea1_repre = self.Get_position_fea1_repre(self.input_S_sent, self.input_S_para,
                                                                  self.input_S_disc)
        S_basic = tf.reshape(tf.concat([self.S_word_repre, self.S_POS_repre, self.S_position_fea1_repre], 2),
                                  shape=[tf.shape(self.input_S_word)[0], -1])
        S_coh = self.Get_S_mem_embedding(S_basic)
        S_repre = tf.concat([S_basic, S_coh], 1)
        return S_repre

    def Get_B_repre(self):
        self.B_word_repre = self.Get_repre_word(self.input_B_word, 'B_word_rnn')
        self.B_POS_repre = self.Get_repre_POS(self.input_B_POS, 'B_POS_rnn')
        self.B_position_fea1_repre = self.Get_position_fea1_repre(self.input_B_sent, self.input_B_para,
                                                                  self.input_B_disc)
        B_basic = tf.reshape(tf.concat([self.B_word_repre, self.B_POS_repre, self.B_position_fea1_repre], 2),
                                  shape=[tf.shape(self.input_B_word)[0], -1])
        B_coh = self.Get_B_mem_embedding(B_basic)
        B_repre = tf.concat([B_basic, B_coh], 1)
        return B_repre

    def Get_action_repre(self, input_):
        """
        get the vector representation of Action History Stack
        we just simple concat each action vector representation
        """
        action_embedding = tf.get_variable('action_embedding', shape=[Flags.Class_size+1, Flags.action_embedding_dim])
        action_embedding_info = tf.nn.embedding_lookup(action_embedding, input_)
        embedding_result = tf.reshape(action_embedding_info, shape=[tf.shape(input_)[0], -1])
        return embedding_result

    def Get_position_fea1_repre(self, sent, para, disc):
        sent_repre = tf.nn.embedding_lookup(self.sent_embedding, sent)
        para_repre = tf.nn.embedding_lookup(self.para_embedding, para)
        disc_repre = tf.nn.embedding_lookup(self.disc_embedding, disc)
        return tf.concat([sent_repre, para_repre, disc_repre], 2)

    def MLP_layer(self, input_):
        '''
        pass the configure_repre into two full connect layer with relu as their activation function
        '''
        # full connect layer parameters
        all_dim = 2*Flags.Stack_edus*(2*Flags.word_rnn_dim+2*Flags.POS_rnn_dim+3*self.position1_embedding_dim) + \
                  2*Flags.Buffer_edus*(2*Flags.word_rnn_dim+2*Flags.POS_rnn_dim+3*self.position1_embedding_dim) + \
                  Flags.Action_actions*(Flags.action_embedding_dim) + \
                  self.same_embedding_dim + self.para_dis_embedding_dim
        FC_W1 = tf.get_variable('Full_Connect_layer_W1', shape=[all_dim, all_dim])
        FC_b1 = tf.get_variable('Full_Connect_layer_b1', shape=[all_dim])
        FC_W2 = tf.get_variable('Full_Connect_layer_W2', shape=[all_dim, all_dim])
        FC_b2 = tf.get_variable('Full_Connect_layer_b2', shape=[all_dim])
        FC1 = tf.nn.relu(tf.matmul(input_, FC_W1)+FC_b1)
        FC2 = tf.nn.relu(tf.matmul(FC1, FC_W2)+FC_b2)
        return FC2

    def get_out(self, input_):
        all_dim = 2*Flags.Stack_edus * (2 * Flags.word_rnn_dim + 2 * Flags.POS_rnn_dim+3*self.position1_embedding_dim) + \
                  2*Flags.Buffer_edus * (2 * Flags.word_rnn_dim + 2 * Flags.POS_rnn_dim+3*self.position1_embedding_dim) + \
                  Flags.Action_actions * (Flags.action_embedding_dim)+ \
                  self.same_embedding_dim + self.para_dis_embedding_dim
        W_out = tf.get_variable('W_out', shape=[all_dim, Flags.Class_size])
        b_out = tf.get_variable('b_out', shape=[Flags.Class_size])
        out = tf.matmul(input_, W_out) + b_out
        return out

    def get_prob(self, input_):
        prob = tf.nn.softmax(input_, -1)
        return prob

    def get_prediction(self, input_):
        predictios = tf.argmax(input_, 1)
        return predictios

    def get_accuracy(self, input_):
        accuracy = tf.cast(tf.equal(self.prediction, tf.argmax(self.gold_action, 1)), "float")
        return accuracy

    def get_entropy_loss(self, input_):
        entropy_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=input_, labels=self.gold_action))
        return entropy_loss