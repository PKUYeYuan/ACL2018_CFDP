#_*_coding=utf-8_*_
#获得词向量字典
word_num = 13794
word_dim = 50
word_to_index = {}
# 处理未登录词，置为全零向量
word_index = 0
unknown = [0.0 for i in range(word_dim)]  # OOV word embedding
word_embedding_matrix = [unknown]  # OOV word at the begin position of embedding matrix
word_to_index['<unk>'] = word_index
word_index += 1

# 处理train中出现的词
word_vector_file = open('../../../glove.6B/glove.6B.50dout.txt', 'r')
lines = word_vector_file.readlines()
for line in lines:
    templine = line.split()
    word = templine[0].lower()
    vector = [float(i) for i in templine[1:]]  
    word_to_index[word] = word_index
    word_embedding_matrix.append(vector)
    word_index += 1
word_vector_file.close()

print('word embedding over\n')

