# coding=utf-8
'''
parameters used in this project
'''
class flags:
    def __init__(self):
        # self.neural_model_setting = 'Basic_model'
        # self.neural_model_setting = 'Basic_model_ex'
        self.neural_model_setting = 'Refine_mem_model'
        # self.data_mode_setting = 'UAS'
        self.data_mode_setting = 'LAS_Fine'
        # self.data_mode_setting = 'LAS_Coarse'
        self.word_vec_dim = 50              # word embedding dimension
        self.vocab_size = 13794             # the size of word vocabulary
        self.edu_padding_length = 20        # padding all edus into 20 length
        self.word_rnn_dim = 200             # EDU word embedding bi-lstm dimension
        self.action_embedding_dim = 50      # action history embedding vector dimension
        self.POS_kinds = 45                 # POS tag kinds
        self.POS_embedding_dim = 15         # POS embedding dimension
        self.POS_rnn_dim = 50               # EDU POS embedding bi-lstm dimension
        self.Stack_edus = 1                 # used edu info numbers in stack
        self.Buffer_edus = 2                # used edu info numbers in buffer
        self.Action_actions = 3             # used action history info numbers in action history stack
        self.learning_rate = 0.001          # optimizer learning rate
        self.stddev_setting = 0.01          # the standard deviation when initializing the variable using truncated normal distribution
        self.model_save_path = './model/'   # path to save NN model
        self.epoch_nums = 15                # epoch numbers

        # class numbers
        if self.data_mode_setting == 'UAS':
            self.Class_size = 4
        elif self.data_mode_setting == 'LAS_Fine':
            self.Class_size = 219
        else:
            self.Class_size = 39
        self.action_history_mode = 'no_relation'
        # self.action_history_mode = 'has_relation'
        # action history mode
        if self.action_history_mode == 'no_relation':
            self.action_history_kinds = 4
        else:
            self.action_history_kinds = self.Class_size
Flags = flags()