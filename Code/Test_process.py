# coding=utf-8
# testing model
import random
import tensorflow as tf
import numpy as np
import pickle as pkl
import datetime
from yy_flags import Flags
from Network_Model import Basic_model, Basic_model_ex, Refine_mem_model
import Parser
def Test(epoch_num):
    pathname = "./model/ATT_GRU_model-"
    print('reading testing data')
    if Flags.data_mode_setting!='LAS_Coarse':
        test_data = pkl.load(open('./tmp_data/fine_test.pkl', 'rb'))
    else:
        test_data = pkl.load(open('./tmp_data/coarse_test.pkl', 'rb'))
    print('test discourse numbers: ', len(test_data))
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
                    m = Basic_model(is_training=False, word_embedding=None)
                elif Flags.neural_model_setting=='Basic_model_ex':
                    m = Basic_model_ex(is_training=False, word_embedding=None)
                else:
                    m = Refine_mem_model(is_training=False, word_embedding=None)

            print('build model over')
            global_step = tf.Variable(0, name="global_step", trainable=False)
            # set learning rate
            # restore model
            names_to_vars = {v.op.name: v for v in tf.global_variables()}
            saver = tf.train.Saver(names_to_vars)
            saver.restore(sess, pathname + str(epoch_num))
            # sess.run(tf.global_variables_initializer())
            arc_eager_parser = Parser.Arc_Eager_Parser()

            all_arcs, uas_correct, las_correct = 0.0, 0.0, 0.0


            time_str = datetime.datetime.now().isoformat()
            print(str(time_str) + ':test epoch ' + str(epoch_num))
            temp_order = list(range(len(test_data)))
            discourse_count = 0
            for discourse_index in temp_order:
                test_discourse = test_data[discourse_index]
                print('test: deal with discourse ' + str(discourse_count))
                discourse_count += 1
                arcs = set(arc_eager_parser.test(test_discourse, sess, m))
                # write the test predict into the result file
                Write_testOut(test_discourse, arcs, epoch_num)

                # calculate uas and las info
                gold_arcs = set([(test_discourse[i][2], i, test_discourse[i][3]) for i in range(len(test_discourse))])
                gold_arcs = sorted(gold_arcs, key=lambda x: x[1])
                temp_arcs = sorted(arcs, key=lambda x: x[1])
                for iter_i in range(len(gold_arcs)):
                    if (temp_arcs[iter_i][0] == gold_arcs[iter_i][0]):
                        uas_correct += 1
                    if (temp_arcs[iter_i][0] == gold_arcs[iter_i][0] and temp_arcs[iter_i][2] == gold_arcs[iter_i][2]):
                        las_correct += 1
                all_arcs += len(gold_arcs)
            print('uas accuracy %d/%d = %f' % (uas_correct, all_arcs, float(uas_correct) / float(all_arcs)))
            print('las accuracy %d/%d = %f' % (las_correct, all_arcs, float(las_correct) / float(all_arcs)))
            # Print_result_test()
            outfile = open("right_ratio.txt", 'a+')
            outfile.write("test: in the iteration " + str(epoch_num) + " the test uas right ratio is: " + str(
            float(uas_correct) / float(all_arcs)) + ' correct_arcs:' + str(uas_correct) + ' all_arcs:' + str(
            all_arcs) + '\n')
            outfile.write("test: in the iteration " + str(epoch_num) + " the test las right ratio is: " + str(
            float(las_correct) / float(all_arcs)) + ' correct_las:' + str(las_correct) + ' all_arcs:' + str(
            all_arcs) + '\n')
            outfile.close()
            return float(uas_correct) / float(all_arcs)

def Write_testOut(discourse, arcs, epoch_num):
    testOutfile = open("./Result/" + "test" + str(epoch_num) + ".txt", 'a+')
    temp_arcs = sorted(arcs, key=lambda x: x[1])
    discourse_length = len(discourse)
    for edu_iter in range(discourse_length):
        edu = discourse[edu_iter]
        outputStr = str(edu_iter + 1) + '|'
        if temp_arcs[edu_iter][1] == edu_iter:
            if temp_arcs[edu_iter][0] == discourse_length:
                outputStr += '0|Root' + '\n'
            else:
                outputStr += str(temp_arcs[edu_iter][0] + 1) + '|' + temp_arcs[edu_iter][2] + '\n'
            testOutfile.write(outputStr)
        else:
            testOutfile.write("didn't predict this edu!ERROR!\n")
    testOutfile.write('\n')
    testOutfile.close()

if __name__=='__main__':
    for i in range(0,7):
        Test(i)