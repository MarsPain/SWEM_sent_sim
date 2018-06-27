# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8') #gb2312
#training the model.
#process--->1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')
import tensorflow as tf
import numpy as np
# from a1_dual_bilstm_cnn_model import DualBilstmCnnModel
from data_utils import create_vocabulary,load_data
import os
import random
# import word2vec
# from weight_boosting import compute_labels_weights,get_weights_for_current_batch,get_weights_label_as_standard_dict,init_weights_dict
#configuration
import gensim
from gensim.models import KeyedVectors
import cPickle

flags=tf.app.flags
flags.DEFINE_string("ckpt_dir","SWEM_1","checkpoint location for the model")
flags.DEFINE_string("model_name","SWEM","which model to use")
flags.DEFINE_string("name_scope","SWEM_1","name scope value.")
flags.DEFINE_string("tokenize_style",'word',"tokenize sentence in char,word,or pinyin.default is char")
flags.DEFINE_string("traning_data_path","data/atec_nlp_sim_train2.csv","path of traning data.")
flags.DEFINE_integer("sentence_len",39,"max sentence length. length should be divide by 3, which is used by k max pooling.") #39
flags.DEFINE_integer("vocab_size",80000,"maximum vocab size.")
flags.DEFINE_integer("batch_size", 200, "Batch size for training/evaluating.")
flags.DEFINE_integer("num_epochs",10,"number of epochs to run.")
flags.DEFINE_boolean("use_pretrained_embedding",False,"whether to use embedding or not.")
flags.DEFINE_float("dropout_keep_prob", 0.5, "dropout keep probability")
FLAGS = tf.app.flags.FLAGS

def train(_):
    vocabulary_word2index, vocabulary_index2word, vocabulary_label2index, vocabulary_index2label= create_vocabulary("data/atec_nlp_sim_train2.csv",FLAGS.vocab_size,
                                                                                              name_scope=FLAGS.name_scope,tokenize_style=FLAGS.tokenize_style)
    vocab_size = len(vocabulary_word2index); print("vocab_size:", vocab_size)
    num_classes = len(vocabulary_index2label); print("num_classes:", num_classes)
    train, valid, test, true_label_percent = load_data(FLAGS.traning_data_path, vocabulary_word2index,
                                                       vocabulary_label2index, FLAGS.sentence_len, FLAGS.name_scope,
                                                       tokenize_style=FLAGS.tokenize_style)
    trainX1, trainX2, trainBlueScores, trainY = train
    validX1, validX2, validBlueScores, validY = valid
    testX1, testX2, testBlueScores, testY = test
    # print("trainX1:", trainX1[0]);print("validX1:", validX1[0]);print("testX1:", testX1[0])
    loadpath = "cache_SWEM_1/train_valid_test.pik"
    x = cPickle.load(open(loadpath, "rb"))
    train, val, test = x[0], x[1], x[2]
    print("trainX1:", train[0]);print("validX1:", val[0]);print("testX1:", test[0])

if __name__=="__main__":
    tf.app.run(train)