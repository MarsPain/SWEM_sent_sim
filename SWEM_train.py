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
import os
GPUID = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUID)
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers
# from tensorflow.contrib import metrics
# from tensorflow.contrib.learn import monitors
from tensorflow.contrib import framework
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.python.platform import tf_logging as logging
# from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec
import cPickle
import numpy as np
import os
import scipy.io as sio
from math import floor
import pdb
from SWEM_model import *
from utils import prepare_data_for_cnn, prepare_data_for_rnn, get_minibatches_idx, normalizing, restore_from_save\
    , _clip_gradients_seperate_norm, tensors_key_in_file, prepare_data_for_emb
# import tempfile
# from tensorflow.examples.tutorials.mnist import input_data

logging.set_verbosity(logging.INFO)
# Basic model parameters as external flags.
# flags = tf.app.flags
# # flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
# FLAGS = flags.FLAGS


class Options(object):
    def __init__(self):
        self.fix_emb = True
        #self.relu_w = True
        #self.reuse_w = False
        self.reuse_cnn = False
        self.reuse_discrimination = True  # reuse cnn for discrimination
        self.restore = True
        self.tanh = False  # activation fun for the top layer of cnn, otherwise relu
        self.model = 'cnn_deconv'  # 'cnn_rnn', 'rnn_rnn' , default: cnn_deconv

        self.permutation = 0
        self.substitution = None  # Deletion(d), Insertion(a), Substitution(s) and Permutation(p)

        self.W_emb = None
        self.cnn_W = None
        self.cnn_b = None
        self.maxlen = 69
        self.n_words = None
        self.filter_shape = 5
        self.filter_size = 300
        self.multiplier = 2  # filtersize multiplier
        self.embed_size = 300
        self.lr = 3e-4
        self.layer = 3
        self.stride = [2, 2, 2]  # for two layer cnn/deconv, use self.stride[0]
        self.batch_size = 200  # 9824
        self.max_epochs = 5000
        self.n_gan = 500  # self.filter_size * 3
        self.L = 100
        self.encoder = 'max'  # 'max' 'concat'
        self.combine_enc = 'mix'
        self.category = 2  # '1' for binary（语句之间的关系分为三种：蕴含、矛盾、中性）

        self.optimizer = 'RMSProp'  # tf.train.AdamOptimizer(beta1=0.9) #'Adam' # 'Momentum' , 'RMSProp'
        self.dropout_ratio = 0.8

        # self.save_path = "./save/snli_emb_10"
        self.save_path = "./save/snli_emb_10"
        self.log_path = "./log"
        self.valid_freq = 100

        # partially use labeled data
        self.part_data = False
        self.portion = 0.01  # 10%  1%

        self.discrimination = False
        self.H_dis = 300

        self.sent_len = self.maxlen + 2 * (self.filter_shape - 1)
        self.sent_len2 = np.int32(floor((self.sent_len - self.filter_shape) / self.stride[0]) + 1)
        self.sent_len3 = np.int32(floor((self.sent_len2 - self.filter_shape) / self.stride[1]) + 1)
        self.sent_len4 = np.int32(floor((self.sent_len3 - self.filter_shape) / self.stride[2]) + 1)

        self.ckpt_dir = "SWEM_1"
        self.model_name = "SWEM"
        self.name_scope = "SWEM_1"
        self.tokenize_style = "word"
        self.traning_data_path = "data/atec_nlp_sim_train2.csv"
        self.sentence_len = 39
        self.vocab_size = 80000
        self.num_epochs = 10
        self.use_pretrained_embedding = True
        self.dropout_keep_prob = 0.5

        print ('Use model %s' % self.model)
        print ('Use %d conv/deconv layers' % self.layer)

    def __iter__(self):
        for attr, value in self.__dict__.iteritems():
            yield attr, value

def auto_encoder(x_1, x_2, x_mask_1, x_mask_2, y, dropout, opt):
    #为何没有如论文中所说引入glove词向量？
    x_1_emb, W_emb = embedding(x_1, opt)  # batch L emb
    #这里要注意为什么x_1_emb和x_2_emb的初始化方法不一样，x_2_emb传入的W_emb参数是
    # x_1_emb生成过程的参数：因为要保证两者同一个词的词向量必须一致？
    x_2_emb = tf.nn.embedding_lookup(W_emb, x_2)

    x_1_emb = tf.nn.dropout(x_1_emb, dropout)  # batch L emb
    x_2_emb = tf.nn.dropout(x_2_emb, dropout)  # batch L emb

    biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
    x_1_emb = layers.fully_connected(tf.squeeze(x_1_emb), num_outputs=opt.embed_size, biases_initializer=biasInit, activation_fn=tf.nn.relu, scope='trans', reuse=None)  # batch L emb
    x_2_emb = layers.fully_connected(tf.squeeze(x_2_emb), num_outputs=opt.embed_size, biases_initializer=biasInit, activation_fn=tf.nn.relu, scope='trans', reuse=True)

    x_1_emb = tf.expand_dims(x_1_emb, 3)  # batch L emb 1
    x_2_emb = tf.expand_dims(x_2_emb, 3)

    if opt.encoder == 'aver':
        H_enc_1 = aver_emb_encoder(x_1_emb, x_mask_1)
        H_enc_2 = aver_emb_encoder(x_2_emb, x_mask_2)

    elif opt.encoder == 'max':
        H_enc_1 = max_emb_encoder(x_1_emb, x_mask_1, opt)
        H_enc_2 = max_emb_encoder(x_2_emb, x_mask_2, opt)

    elif opt.encoder == 'concat':
        H_enc_1 = concat_emb_encoder(x_1_emb, x_mask_1, opt)
        H_enc_2 = concat_emb_encoder(x_2_emb, x_mask_2, opt)

    # discriminative loss term
    if opt.combine_enc == 'mult':
        H_enc = tf.multiply(H_enc_1, H_enc_2)  # batch * n_gan

    if opt.combine_enc == 'concat':
        H_enc = tf.concat([H_enc_1, H_enc_2], 1)

    if opt.combine_enc == 'sub':
        H_enc = tf.subtract(H_enc_1, H_enc_2)

    if opt.combine_enc == 'mix':
        H_1 = tf.multiply(H_enc_1, H_enc_2)
        H_2 = tf.concat([H_enc_1, H_enc_2], 1)
        H_3 = tf.subtract(H_enc_1, H_enc_2)
        H_enc = tf.concat([H_1, H_2, H_3], 1)

    # calculate the accuracy
    logits = discriminator_2layer(H_enc, opt, dropout, prefix='classify_', num_outputs=opt.category, is_reuse=None)
    prob = tf.nn.softmax(logits)

    correct_prediction = tf.equal(tf.argmax(prob, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))

    train_op = layers.optimize_loss(
        loss,
        framework.get_global_step(),
        optimizer='Adam',
        # variables=d_vars,
        learning_rate=opt.lr)

    return accuracy, loss, train_op, W_emb, prob

def main():
    opt = Options()
    vocabulary_word2index, vocabulary_index2word, vocabulary_label2index, vocabulary_index2label = create_vocabulary(
        "data/atec_nlp_sim_train2.csv", opt.vocab_size,
        name_scope=opt.name_scope, tokenize_style=opt.tokenize_style)
    vocab_size = len(vocabulary_word2index);
    print("vocab_size:", vocab_size)
    num_classes = len(vocabulary_index2label);
    print("num_classes:", num_classes)
    train, valid, test, true_label_percent = load_data(opt.traning_data_path, vocabulary_word2index,
                                                       vocabulary_label2index, opt.sentence_len, opt.name_scope,
                                                       tokenize_style=opt.tokenize_style)
    train_q, train_a, _, train_lab = train
    val_q, val_a, _, val_lab = valid
    test_q, test_a, _, test_lab = test
    wordtoix = vocabulary_word2index; ixtoword = vocabulary_index2word

    opt.n_words = len(ixtoword)
    # loadpath = "./data/snli.p"
    # x = cPickle.load(open(loadpath, "rb"))
    #
    # train, val, test = x[0], x[1], x[2]
    # wordtoix, ixtoword = x[4], x[5]
    #
    # train_q, train_a, train_lab = train[0], train[1], train[2]
    # val_q, val_a, val_lab = val[0], val[1], val[2]
    # test_q, test_a, test_lab = test[0], test[1], test[2]
    #
    # train_lab = np.array(train_lab, dtype='float32')
    # val_lab = np.array(val_lab, dtype='float32')
    # test_lab = np.array(test_lab, dtype='float32')
    #
    # opt = Options()
    # opt.n_words = len(ixtoword)
    #
    # del x

    print(dict(opt))
    print('Total words: %d' % opt.n_words)

    #若partially use labeled data则进行以下操作，这部分操作什么意思？
    # 目前猜测part_data设置为True时只利用部分训练集，portion就是保留的训练集大小,应该是用于测试模型阶段使用的
    if opt.part_data:
        np.random.seed(123)
        train_ind = np.random.choice(len(train_q), int(len(train_q)*opt.portion), replace=False)
        train_q = [train_q[t] for t in train_ind]
        train_a = [train_a[t] for t in train_ind]
        train_lab = [train_lab[t] for t in train_ind]
    #验证训练集和预处理好的词嵌入文件是否对齐
    try:
        params = np.load('./data/snli_emb.p')
        if params[0].shape == (opt.n_words, opt.embed_size):
            print('Use saved embedding.')
            #pdb.set_trace()
            opt.W_emb = np.array(params[0], dtype='float32')
        else:
            print('Emb Dimension mismatch: param_g.npz:' + str(params[0].shape) + ' opt: ' + str(
                (opt.n_words, opt.embed_size)))
            opt.fix_emb = False
    except IOError:
        print('No embedding file found.')
        opt.fix_emb = False

    with tf.device('/gpu:1'):
        #注意训练数据是两批句子，所以x的占位符要成对定义
        x_1_ = tf.placeholder(tf.int32, shape=[opt.batch_size, opt.maxlen])
        x_2_ = tf.placeholder(tf.int32, shape=[opt.batch_size, opt.maxlen])
        x_mask_1_ = tf.placeholder(tf.float32, shape=[opt.batch_size, opt.maxlen])
        x_mask_2_ = tf.placeholder(tf.float32, shape=[opt.batch_size, opt.maxlen])
        y_ = tf.placeholder(tf.float32, shape=[opt.batch_size, opt.category])
        keep_prob = tf.placeholder(tf.float32)
        #auto_encoder就是模型的定义、模型运行过程中的所有tensor，这个项目将其封装起来了，很值得借鉴的工程技巧
        # 返回的是一些重要的tensor，后面sess.run的时候作为参数传入
        accuracy_, loss_, train_op_, W_emb, logits_ = auto_encoder(x_1_, x_2_, x_mask_1_, x_mask_2_, y_, keep_prob, opt)
        merged = tf.summary.merge_all()

    def do_eval(sess, train_q, train_a, train_lab):
        train_correct = 0.0
        # number_examples = len(train_q)
        # print("valid examples:", number_examples)
        eval_loss, eval_accc, eval_counter = 0.0, 0.0, 0
        eval_true_positive, eval_false_positive, eval_true_negative, eval_false_negative = 0, 0, 0, 0
        # batch_size = 1
        kf_train = get_minibatches_idx(len(train_q), opt.batch_size, shuffle=True)
        for _, train_index in kf_train:
            train_sents_1 = [train_q[t] for t in train_index]
            train_sents_2 = [train_a[t] for t in train_index]
            train_labels = [train_lab[t] for t in train_index]
            train_labels = np.array(train_labels)
            # print("train_labels", train_labels.shape)
            # train_labels = train_labels.reshape((len(train_labels), opt.category))
            train_labels = np.eye(opt.category)[train_labels]
            x_train_batch_1, x_train_mask_1 = prepare_data_for_emb(train_sents_1, opt)
            x_train_batch_2, x_train_mask_2 = prepare_data_for_emb(train_sents_2, opt)

            curr_eval_loss, curr_accc, logits = sess.run([loss_, accuracy_, logits_],feed_dict={x_1_: x_train_batch_1, x_2_: x_train_batch_2, x_mask_1_: x_train_mask_1, x_mask_2_: x_train_mask_2,
                                                           y_: train_labels, keep_prob: 1.0})
            true_positive, false_positive, true_negative, false_negative = compute_confuse_matrix(logits, train_labels)  # logits:[batch_size,label_size]-->logits[0]:[label_size]
            # write_predict_error_to_file(start,file_object,logits[0], evalY[start:end][0],vocabulary_index2word,evalX1[start:end],evalX2[start:end])
            eval_loss, eval_accc, eval_counter = eval_loss + curr_eval_loss, eval_accc + curr_accc, eval_counter + 1  # 注意这里计算loss和accc的方法，计算累加值，然后归一化
            eval_true_positive, eval_false_positive = eval_true_positive + true_positive, eval_false_positive + false_positive
            eval_true_negative, eval_false_negative = eval_true_negative + true_negative, eval_false_negative + false_negative
            # weights_label = compute_labels_weights(weights_label, logits, evalY[start:end]) #compute_labels_weights(weights_label,logits,labels)
        print("true_positive:", eval_true_positive, ";false_positive:", eval_false_positive, ";true_negative:",
              eval_true_negative, ";false_negative:", eval_false_negative)
        p = float(eval_true_positive) / float(eval_true_positive + eval_false_positive+1)
        r = float(eval_true_positive) / float(eval_true_positive + eval_false_negative+1)
        f1_score = (2 * p * r) / (p + r + 1)
        print("eval_counter:", eval_counter, ";eval_acc:", eval_accc)
        return eval_loss / float(eval_counter), eval_accc / float(eval_counter), f1_score, p, r

    uidx = 0
    max_val_accuracy = 0.
    max_test_accuracy = 0.
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    np.set_printoptions(precision=3)
    np.set_printoptions(threshold=np.inf)
    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(opt.log_path + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(opt.log_path + '/test', sess.graph)
        sess.run(tf.global_variables_initializer())
        if opt.restore: #若使用已保存好的参数
            try:
                #pdb.set_trace()
                t_vars = tf.trainable_variables()
                # print([var.name[:-2] for var in t_vars])
                save_keys = tensors_key_in_file(opt.save_path)

                # pdb.set_trace()
                # print(save_keys.keys())
                ss = set([var.name for var in t_vars]) & set([s + ":0" for s in save_keys.keys()])
                cc = {var.name: var for var in t_vars}
                #pdb.set_trace()

                # only restore variables with correct shape
                ss_right_shape = set([s for s in ss if cc[s].get_shape() == save_keys[s[:-2]]])

                loader = tf.train.Saver(var_list=[var for var in t_vars if var.name in ss_right_shape])
                loader.restore(sess, opt.save_path)

                print("Loading variables from '%s'." % opt.save_path)
                print("Loaded variables:" + str(ss))

            except:
                print("No saving session, using random initialization")
                sess.run(tf.global_variables_initializer())

        try:
            best_acc = 2.0; best_f1_score = 2.0
            for epoch in range(opt.max_epochs):
                print("Starting epoch %d" % epoch)
                loss, acc = 0.0, 0.0
                kf = get_minibatches_idx(len(train_q), opt.batch_size, shuffle=True)    #随机创建minibatch数据
                for _, train_index in kf:
                    uidx += 1
                    sents_1 = [train_q[t] for t in train_index] #根据索引回到总数据集中寻找相应数据
                    sents_2 = [train_a[t] for t in train_index]
                    x_labels = [train_lab[t] for t in train_index]
                    x_labels = np.array(x_labels)
                    # print("x_labels:", x_labels.shape)
                    # 为何要在这里进行reshape,是想进行onehot操作？但是这明显是错误的，((len(x_labels),))怎么能reshape成((len(x_labels),opt.category))
                    # x_labels = x_labels.reshape((len(x_labels),opt.category))
                    # one-hot向量化
                    x_labels = np.eye(opt.category)[x_labels]

                    #prepare_data_for_emb函数的作用是什么?初步猜测是把sents中每一个单词替换成相应的索引，然后才能根据索引获取词向量
                    x_batch_1, x_batch_mask_1 = prepare_data_for_emb(sents_1, opt)
                    x_batch_2, x_batch_mask_2 = prepare_data_for_emb(sents_2, opt)

                    _, curr_loss, curr_accuracy = sess.run([train_op_, loss_, accuracy_], feed_dict={x_1_: x_batch_1, x_2_: x_batch_2,
                                       x_mask_1_: x_batch_mask_1, x_mask_2_: x_batch_mask_2, y_: x_labels, keep_prob: opt.dropout_ratio})
                    loss, acc = loss + curr_loss, acc + curr_accuracy
                    if uidx % 100 == 0:
                        print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tAcc:%.3f\t" % (epoch, uidx, loss/float(uidx), acc/float(uidx)))

                    if uidx % opt.valid_freq == 0:
                        # do_eval参数待修改
                        eval_loss, eval_accc, f1_scoree, precision, recall = do_eval(sess, train_q, train_a, train_lab)
                        # weights_dict = get_weights_label_as_standard_dict(weights_label)
                        # print("label accuracy(used for label weight):==========>>>>", weights_dict)
                        print("【Validation】Epoch %d\t Loss:%.3f\tAcc %.3f\tF1 Score:%.3f\tPrecision:%.3f\tRecall:%.3f" % (
                            epoch, eval_loss, eval_accc, f1_scoree, precision, recall))
                        # save model to checkpoint
                        if eval_accc > best_acc and f1_scoree > best_f1_score:
                            save_path = opt.ckpt_dir + "/model.ckpt"
                            print("going to save model. eval_f1_score:", f1_scoree, ";previous best f1 score:", best_f1_score,
                                  ";eval_acc", str(eval_accc), ";previous best_acc:", str(best_acc))
                            saver.save(sess, save_path, global_step=epoch)
                            best_acc = eval_accc
                            best_f1_score = f1_scoree

                    #每训练valid_freq个minibatch就在训练集、验证集和测试集上计算准确率，并更新最优测试集准确率
            #         if uidx % opt.valid_freq == 0:
            #             train_correct = 0.0
            #             kf_train = get_minibatches_idx(len(train_q), opt.batch_size, shuffle=True)
            #             for _, train_index in kf_train:
            #                 train_sents_1 = [train_q[t] for t in train_index]
            #                 train_sents_2 = [train_a[t] for t in train_index]
            #                 train_labels = [train_lab[t] for t in train_index]
            #                 train_labels = np.array(train_labels)
            #                 # print("train_labels", train_labels.shape)
            #                 # train_labels = train_labels.reshape((len(train_labels), opt.category))
            #                 train_labels = np.eye(opt.category)[train_labels]
            #                 x_train_batch_1, x_train_mask_1 = prepare_data_for_emb(train_sents_1, opt)
            #                 x_train_batch_2, x_train_mask_2 = prepare_data_for_emb(train_sents_2, opt)
            #
            #                 train_accuracy = sess.run(accuracy_,
            #                                           feed_dict={x_1_: x_train_batch_1, x_2_: x_train_batch_2, x_mask_1_: x_train_mask_1, x_mask_2_: x_train_mask_2,
            #                                                      y_: train_labels, keep_prob: 1.0})
            #
            #                 train_correct += train_accuracy * len(train_index)
            #
            #             train_accuracy = train_correct / len(train_q)
            #
            #             # print("Iteration %d: Training loss %f, dis loss %f, rec loss %f" % (uidx,
            #             #                                                                     loss, dis_loss, rec_loss))
            #             print("Train accuracy %f " % train_accuracy)
            #
            #             val_correct = 0.0
            #             is_train = True
            #             kf_val = get_minibatches_idx(len(val_q), opt.batch_size, shuffle=True)
            #             for _, val_index in kf_val:
            #                 val_sents_1 = [val_q[t] for t in val_index]
            #                 val_sents_2 = [val_a[t] for t in val_index]
            #                 val_labels = [val_lab[t] for t in val_index]
            #                 val_labels = np.array(val_labels)
            #                 # val_labels = val_labels.reshape((len(val_labels), opt.category))
            #                 val_labels = np.eye(opt.category)[val_labels]
            #                 x_val_batch_1, x_val_mask_1 = prepare_data_for_emb(val_sents_1, opt)
            #                 x_val_batch_2, x_val_mask_2 = prepare_data_for_emb(val_sents_2, opt)
            #
            #                 val_accuracy = sess.run(accuracy_, feed_dict={x_1_: x_val_batch_1, x_2_: x_val_batch_2,
            #                                                               x_mask_1_: x_val_mask_1, x_mask_2_: x_val_mask_2, y_: val_labels, keep_prob: 1.0})
            #
            #                 val_correct += val_accuracy * len(val_index)
            #
            #             val_accuracy = val_correct / len(val_q)
            #
            #             print("Validation accuracy %f " % val_accuracy)
            #
            #             if val_accuracy > max_val_accuracy:
            #                 max_val_accuracy = val_accuracy
            #
            #                 test_correct = 0.0
            #                 kf_test = get_minibatches_idx(len(test_q), opt.batch_size, shuffle=True)
            #                 for _, test_index in kf_test:
            #                     test_sents_1 = [test_q[t] for t in test_index]
            #                     test_sents_2 = [test_a[t] for t in test_index]
            #                     test_labels = [test_lab[t] for t in test_index]
            #                     test_labels = np.array(test_labels)
            #                     # test_labels = test_labels.reshape((len(test_labels), opt.category))
            #                     test_labels = np.eye(opt.category)[test_labels]
            #                     x_test_batch_1, x_test_mask_1 = prepare_data_for_emb(test_sents_1, opt)
            #                     x_test_batch_2, x_test_mask_2 = prepare_data_for_emb(test_sents_2, opt)
            #
            #                     test_accuracy = sess.run(accuracy_, feed_dict={x_1_: x_test_batch_1, x_2_: x_test_batch_2,
            #                                                                    x_mask_1_: x_test_mask_1, x_mask_2_: x_test_mask_2,
            #                                                                    y_: test_labels, keep_prob: 1.0})
            #
            #                     test_correct += test_accuracy * len(test_index)
            #
            #                 test_accuracy = test_correct / len(test_q)
            #
            #                 print("Test accuracy %f " % test_accuracy)
            #
            #                 max_test_accuracy = test_accuracy
            #
            #     print("Epoch %d: Max Test accuracy %f" % (epoch, max_test_accuracy))
            #
            # print("Max Test accuracy %f " % max_test_accuracy)

        except KeyboardInterrupt:
            print('Training interupted')
            print("Max Test accuracy %f " % max_test_accuracy)

def compute_confuse_matrix(logit, label):
    """
    compoute f1_score.
    :param logits: [batch_size,label_size]
    :param evalY: [batch_size,label_size]
    :return:
    """
    # print("logit:", logit, "label", label)
    # predict=np.argmax(logit)
    # true_positive=0  #TP:if label is true('1'), and predict is true('1')
    # false_positive=0 #FP:if label is false('0'),but predict is ture('1')
    # true_negative=0  #TN:if label is false('0'),and predict is false('0')
    # false_negative=0 #FN:if label is false('0'),but predict is true('1')
    # if predict==1 and label==1:
    #     true_positive=1
    # elif predict==1 and label==0:
    #     false_positive=1
    # elif predict==0 and label==0:
    #     true_negative=1
    # elif predict==0 and label==1:
    #     false_negative=1
    length = len(logit)
    for i in range(length):
        predict = np.argmax(logit[i]); true_label = np.argmax(label[0])
        # print(predict, true_label)
        true_positive = 0  # TP:if label is true('1'), and predict is true('1')
        false_positive = 0  # FP:if label is false('0'),but predict is ture('1')
        true_negative = 0  # TN:if label is false('0'),and predict is false('0')
        false_negative = 0  # FN:if label is false('0'),but predict is true('1')
        if predict == 1 and true_label == 1:
            true_positive = 1
        elif predict == 1 and true_label == 0:
            false_positive = 1
        elif predict == 0 and true_label == 0:
            true_negative = 1
        elif predict == 0 and true_label == 1:
            false_negative = 1

    return true_positive,false_positive,true_negative,false_negative

if __name__ == '__main__':
    main()