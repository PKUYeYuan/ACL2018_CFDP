# coding=utf-8
'''
this file converts the original data file into an intermediate format(such as numpy in our work)
'''
import numpy as np
import pickle as pkl
from yy_flags import Flags
def convert_word_vec(ori_dir, out_dir):
    '''
    converts the word vector file glove_50d.txt into vec.npy
    and get the word2id info
    :param ori_dir: directory of original word vector file
    :param out_dir: directory of output word vector file
    :return: word2id dictionary
    '''
    # total words is 13794
    infile_name = ori_dir+'glove_50d.txt'
    outfile_name = out_dir+'vec.npy'
    word2id = {}
    # set the unknown word vec to zeros vector
    word_id = 0
    unknown = [0.0 for i in range(Flags.word_vec_dim)]  # OOV word embedding
    word_embedding_matrix = [unknown]  # OOV word at the begin position of embedding matrix
    word2id['<unk>'] = word_id
    word_id += 1

    # deal with the word that appears in the training set
    infile = open(infile_name, 'r')
    lines = infile.readlines()
    for line in lines:
        templine = line.split()
        word = templine[0].lower()
        vector = [float(i) for i in templine[1:]]
        word2id[word] = word_id
        word_embedding_matrix.append(vector)
        word_id += 1
    infile.close()

    word_embedding_matrix = np.array(word_embedding_matrix, dtype=np.float32)
    np.save(outfile_name, word_embedding_matrix)
    print('word embedding over', len(word_embedding_matrix))
    pkl.dump(word2id, open(out_dir+'word2id.pkl', 'wb'))

def convert_pos2id(ori_dir, out_dir):
    '''
    get the pos2id info
    '''
    pos2id = {}
    infile_name = ori_dir+'pos2id.txt'
    infile = open(infile_name, 'r')
    lines = infile.readlines()
    for line in lines:
        raw = line.strip().split()
        pos2id[raw[0]] = int(raw[1])
    infile.close()
    pkl.dump(pos2id, open(out_dir+'pos2id.pkl', 'wb'))

def convert_data_file(ori_dir, out_dir):
    '''
    convert the training, validation and testing original data file into pkl mode
    and change the word and pos into their corresponding id in word2id and pos2id
    '''
    word2id = pkl.load(open('./tmp_data/word2id.pkl', 'rb'))
    pos2id = pkl.load(open('./tmp_data/pos2id.pkl', 'rb'))
    def f(filename):
        '''
        convert single original file into pkl mode
        '''
        infilename = ori_dir+filename
        outfilename = out_dir + filename + '.pkl'
        out_content = []
        single_discourse_content = []
        infile = open(infilename, 'r')
        lines = infile.readlines()
        for line in lines:
            if line.strip()=='': #save the previous discourse info
                # change the id of root node to number of edus in a discourse
                # (just simply put the root node into the last positoin of B(Buffer))
                discourse_edu_nums = len(single_discourse_content)
                single_discourse_content_final_result = []
                for tmp in single_discourse_content:
                    if tmp[2]<0: # if the head node is ROOT
                        tmp_tuple = (tmp[0], tmp[1], discourse_edu_nums, tmp[3], tmp[4],
                                     tmp[5], tmp[6], tmp[7], tmp[8], tmp[9])
                        single_discourse_content_final_result.append(tmp_tuple)
                    else:
                        single_discourse_content_final_result.append(tmp)
                out_content.append(single_discourse_content_final_result)
                single_discourse_content = []
                continue

            raw = line.strip().split('|')
            id = raw[0]                         #EDU id in discourse
            EDU_word = raw[1].lower().split()   # EDU word info
            EDU_word_id_origi = [word2id[word] if word in word2id.keys() else 0 for word in EDU_word]
            EDU_word_id_padding = [0 for i in range(Flags.edu_padding_length)]
            for i in range(min(len(EDU_word_id_origi), Flags.edu_padding_length)):
                EDU_word_id_padding[i] = EDU_word_id_origi[i]

            EDU_POS = raw[2].split()            # EDU POS info of each word
            EDU_POS_id_origi = [pos2id[POS] if POS in pos2id.keys() else 0 for POS in EDU_POS]
            EDU_POS_id_padding = [0 for i in range(Flags.edu_padding_length)]
            for i in range(min(len(EDU_POS_id_origi), Flags.edu_padding_length)):
                EDU_POS_id_padding[i] = EDU_POS_id_origi[i]
            head = int(raw[3])-1                # head info of an EDU
            label = raw[4]                      # relation label between current EDU and its head EDU
            sent_id = int(raw[5])               # sentence id(1,2,later)
            EDU_pos_in_sent = int(raw[6])       # EDU position in sentence(1,2,later)
            EDU_pos_in_discourse = int(raw[7])  # EDU position in discourse(1,2,later)
            parag_id = int(raw[8])              # paragraph id(1,2,later)
            EDU_pos_in_parag = int(raw[9])      # EDU position in paragraph(1,2,later)
            EDU_id_in_para = int(raw[10])       # EDU index in paragraph(1,2,3,...)
            EDU_info = (EDU_word_id_padding, EDU_POS_id_padding, head, label,
                        sent_id, EDU_pos_in_sent, EDU_pos_in_discourse, parag_id,
                        EDU_pos_in_parag, EDU_id_in_para)
            single_discourse_content.append(EDU_info)
        pkl.dump(out_content, open(outfilename, 'wb'))
        print('dump '+outfilename)
    # convert training files
    f('coarse_train')
    f('fine_train')
    # convert validation(development) files
    f('coarse_dev')
    f('fine_dev')
    # convert testing files
    f('coarse_test')
    f('fine_test')

def get_action2id():
    UAS_action2id = {"SHIFT_": 0, 'RIGHT_': 1, 'LEFT_': 2, 'REDUCE_': 3}
    pkl.dump(UAS_action2id, open('./tmp_data/UAS_action2id.pkl', 'wb'))

    LAS_Fine_action2id = {'SHIFT_':0, 'REDUCE_':1}
    fine_relation_list = pkl.load(open('./tmp_data/fine_relation_list.pkl', 'rb'))
    fine_id = 2
    for transition in ['LEFT', 'RIGHT']:
        for fine_relation in fine_relation_list:
            if transition=='RIGHT' and fine_relation=='Root':
                continue
            LAS_Fine_action2id[transition+'_'+fine_relation] = fine_id
            fine_id += 1
    pkl.dump(LAS_Fine_action2id, open('./tmp_data/LAS_Fine_action2id.pkl', 'wb'))
    LAS_Coarse_action2id = {'SHIFT_':0, 'REDUCE_':1}
    coarse_relation_list = pkl.load(open('./tmp_data/coarse_relation_list.pkl', 'rb'))
    coarse_id = 2
    for transition in ['LEFT', 'RIGHT']:
        for coarse_relation in coarse_relation_list:
            if transition=='RIGHT' and coarse_relation=='Root':
                continue
            LAS_Coarse_action2id[transition+'_'+coarse_relation] = coarse_id
            coarse_id += 1
    pkl.dump(LAS_Coarse_action2id, open('./tmp_data/LAS_Coarse_action2id.pkl', 'wb'))
    # print((LAS_Coarse_action2id))

def get_fine_relation_list():
    fine_relation_list = []
    # test_data = pkl.load(open('./tmp_data/fine_test.pkl', 'rb'))
    # for discourse in test_data:
    #     for edu in discourse:
    #         fine_relation_list.append(edu[3])
    test_data = pkl.load(open('./tmp_data/coarse_train.pkl', 'rb'))
    for discourse in test_data:
        for edu in discourse:
            fine_relation_list.append(edu[3])
    # test_data = pkl.load(open('./tmp_data/fine_dev.pkl', 'rb'))
    # for discourse in test_data:
    #     for edu in discourse:
    #         fine_relation_list.append(edu[3])
    # print((set(fine_relation_list)))
def convert_main():
    original_data_dir = '../Data/'
    original_wordvec_dir = '../Word_Vec/'
    output_dir = './tmp_data/'
    convert_word_vec(original_wordvec_dir, output_dir)
    convert_pos2id(original_data_dir, output_dir)
    convert_data_file(original_data_dir, output_dir)

if __name__=='__main__':
    convert_main()
    get_action2id()
    get_fine_relation_list()