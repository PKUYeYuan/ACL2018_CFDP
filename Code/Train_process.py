# coding=utf-8
import random
import tensorflow as tf
import numpy as np
import pickle as pkl
import datetime
from yy_flags import Flags
from Network_Model import Basic_model, Basic_model_ex, Refine_mem_model
import Parser
from Test_process import Test
from Dev_process import Dev
# training the network model based on the arc-eager parser
def Train():
    print('reading word embedding from ./data/vec.npy')
    word_embedding = np.load('./tmp_data/vec.npy')
    print('reading training data')
    if Flags.data_mode_setting!='LAS_Coarse':
        train_data = pkl.load(open('./tmp_data/fine_train.pkl', 'rb'))
    else:
        train_data = pkl.load(open('./tmp_data/coarse_train.pkl', 'rb'))
    print('train discourse numbers: ', len(train_data))
    with tf.Graph().as_default():
        # config GPU mode of TensorFlow
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        with sess.as_default():
            initializer = tf.truncated_normal_initializer(mean=0.0, stddev=Flags.stddev_setting)
            print('build model begin')
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                if Flags.neural_model_setting=='Basic_model':
                    m = Basic_model(is_training=True, word_embedding=word_embedding)
                elif Flags.neural_model_setting=='Basic_model_ex':
                    m = Basic_model_ex(is_training=True, word_embedding=word_embedding)
                else:
                    m = Refine_mem_model(is_training=True, word_embedding=word_embedding)
            print('build model over')
            global_step = tf.Variable(0, name="global_step", trainable=False)
            # set learning rate
            learning_rate = Flags.learning_rate
            # TODO try other Optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(m.en_loss, global_step=global_step)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=10)
            arc_eager_parser = Parser.Arc_Eager_Parser()
            for one_epoch in range(Flags.epoch_nums):
                correct_num, all_num = 0.0, 0.0
                time_str = datetime.datetime.now().isoformat()
                print(str(time_str)+': epoch '+str(one_epoch)+' starts')
                temp_order = list(range(len(train_data)))
                random.shuffle(temp_order)
                discourse_count = 0
                for discourse_index in temp_order:
                    train_discourse = train_data[discourse_index]
                    print('deal with discourse '+str(discourse_count))
                    discourse_count += 1
                    correct, all = arc_eager_parser.train(train_discourse, sess, m, train_op)
                    correct_num += correct
                    all_num += all
                print('acc', correct_num/all_num)
                print('##################saving model#######################')
                path = saver.save(sess, Flags.model_save_path + 'ATT_GRU_model-'+str(one_epoch))
                time_str = datetime.datetime.now().isoformat()
                tempstr = "{}: have saved model to ".format(time_str) + path
                print(tempstr)
                Dev(one_epoch)
                Test(one_epoch)


if __name__=='__main__':
    Train()