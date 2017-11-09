import tensorflow as tf
import argparse
from c_util import *
import numpy as np
import numpy as np
import os
import random

class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, epsilon=1e-5, momentum = 0.1, name="batch_norm"):
        with tf.variable_scope(name) as scope:
            self.epsilon = epsilon
            self.momentum = momentum

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name=name
    
    def __call__(self, x, train=True, reuse=False):
        shape = x.get_shape().as_list()

        with tf.variable_scope(self.name) as scope:
            self.gamma = tf.get_variable("gamma", [shape[-1]],
            initializer=tf.random_normal_initializer(1., 0.02))
            self.beta = tf.get_variable("beta", [shape[-1]],
            initializer=tf.constant_initializer(0.))

            mean, variance = tf.nn.moments(x, [0, 1])

            return tf.nn.batch_norm_with_global_normalization(
                     x, mean, variance, self.beta, self.gamma, self.epsilon,
                     scale_after_normalization=True)

class model():
    def __init__(self, args):
        self.args = args
        
        #単語全体のbyte数 単語数 文字の位置
        self.feature = tf.placeholder(dtype=tf.float32, shape=[None, args.max_time_step, 3])
        self.inputs = tf.placeholder(dtype=tf.int32, shape=[None, self.args.max_time_step, self.args.max_word_length])
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None, args.max_time_step, 17])
        
        with tf.variable_scope('Embedding_and_Conv') as scope:
            cnn_outputs = []
            
            embedding_weight = tf.Variable(tf.random_uniform([self.args.vocab_size, self.args.embedding_size],-1.,1.), name='embedding_weight')
            word_t  = tf.split(self.inputs, self.args.max_time_step, axis=1)

            for t in range(self.args.max_time_step):
                char_index = tf.reshape(word_t[t], shape=[-1, self.args.max_word_length])
                embedded = tf.nn.embedding_lookup(embedding_weight, char_index)
                embedded_ = tf.expand_dims(embedded, -1)

                t_cnn_outputs = []
                for i, (kernel, filter_num) in enumerate(zip(self.args.kernels, self.args.filter_nums)):
                    bn = batch_norm(name='batch_norm_{}_{}'.format(t,i))
                    conv_ = tf.layers.conv2d(embedded_, filter_num, kernel_size=[kernel, self.args.embedding_size], padding="valid", strides=[1, 1], activation=tf.nn.relu, name="conv_{}".format(kernel), reuse= True if t!=0 else False)
                    pool_ = tf.layers.max_pooling2d(conv_, pool_size=[self.args.max_word_length-kernel+1, 1], strides=[1, 1])
                    t_cnn_outputs.append(tf.reshape(bn(pool_), (-1, filter_num)))

                cnn_output = tf.contrib.layers.batch_norm(tf.concat([t_cnn_output for t_cnn_output in t_cnn_outputs], axis=-1))
                higway_out = self.highway(cnn_output, sum(self.args.filter_nums), reuse= True if t!=0 else False) if  self.args.highway == True \
                                                                                            else cnn_output
                cnn_outputs.append(tf.concat([higway_out, self.feature[:,t,:]], axis=-1))
            cnn_outputs = tf.convert_to_tensor(cnn_outputs)
     
    
        if self.args.cell_model == 'rnn':
            cell_fn = tf.contrib.rnn.BasicRNNCell
        elif self.args.cell_model == 'gru':
            cell_fn = tf.contrib.rnn.GRUCell
        elif self.args.cell_model == 'lstm':
            cell_fn = tf.contrib.rnn.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(self.args.cell_model))

        with tf.variable_scope('RNN') as scope:
            def cell():
                cell_ = cell_fn(self.args.rnn_size, reuse=tf.get_variable_scope().reuse)
                if self.args.keep_prob < 1.:
                    cell_ = tf.contrib.rnn.DropoutWrapper(cell_, output_keep_prob=self.args.keep_prob)
                return cell_

            cell = tf.contrib.rnn.MultiRNNCell([cell() for _ in range(self.args.num_layers)], state_is_tuple = True)
            state_in = cell.zero_state(batch_size=self.args.batch_size, dtype=tf.float32)

            rnn_out, self.out_state = tf.nn.dynamic_rnn(cell, cnn_outputs, initial_state=state_in, time_major=True,dtype=tf.float32)
        
        with tf.variable_scope("dense_layer") as scope:
            logits = []
            outs = []
            for t in range(self.args.max_time_step):
                if t != 0:
                    tf.get_variable_scope().reuse_variables()

                logit = tf.layers.dense(rnn_out[t], 17, name="Dense")
                out = tf.nn.softmax(logit)
                logits.append(logit)
                outs.append(out)

            logits = tf.convert_to_tensor(logits)
            self.outs = tf.transpose(tf.convert_to_tensor(outs), (1,0,2))

        with tf.variable_scope("loss") as scope:
            logits = tf.reshape(logits, shape=(-1, 17))
            labels = tf.reshape(self.labels, shape=(-1, 17))
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    
    def highway(self, x, size, activation=tf.nn.tanh, carry_bias=-1.0, reuse=False):
        T = tf.layers.dense(x, size, activation=tf.nn.sigmoid, name="transfort_gate", reuse=reuse)
        H = tf.layers.dense(x, size, activation=activation, name="activation", reuse=reuse)
        C = tf.subtract(1.0, T, name="carry_gate")

        y = tf.add(tf.multiply(H, T), tf.multiply(x, C), "y")
        return y
    
    def train(self):
        opt_ = tf.train.AdamOptimizer(self.args.lr).minimize(self.loss)
        
        if self.args.test:
            train_func, test_func = mk_train_func("../data/train_before_sentence_.txt", "../data/train_class_.txt", "../data/char_dict.txt", "../data/class_dict.txt", self.args.batch_size, self.args.max_time_step, self.args.max_word_length, p=0.1)
        else:
            train_func = mk_train_func("../data/train_before_sentence_.txt", "../data/train_class_.txt", "../data/char_dict.txt", "../data/class_dict.txt", selfargs.batch_size, self.args.max_time_step, self.args.max_word_length, False)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            graph = tf.summary.FileWriter("./logs", sess.graph)
            
            for itr, (data, feature, label) in enumerate(train_func()):
                loss, _ = sess.run([self.loss, opt_], {self.feature:feature, self.inputs:data, self.labels:label})

                if itr % 10 == 0:
                    print(itr, "  :", loss)
            
                if itr % 1000 == 0:
                    saver.save(sess, 'save/model.ckpt', itr)
                    print('-----------------------saved model--------------------------')
                
                if itr == self.args.itrs:
                    break

            if self.args.test:
                result = []
                for i, (data, feature, label, choiced_s, choiced_l) in enumerate(test_func()):
                    try:
                        output_ = sess.run(self.outs, {self.feature:feature, self.inputs:data})
                        result.append(mk_score_board(args, choiced_s, choiced_l, output_, label))
                    except:
                        break
                
                #print(result)
                to_csv(result, "test.csv")    

    def test(self):
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--lr", dest="lr", type=float, default= 0.002)
    parser.add_argument("--cell_model", dest="cell_model", type= str, default="gru")
    parser.add_argument("--data_dir", dest="data_dir", type=str, default="../data/")
    parser.add_argument("--num_layers", dest="num_layers", type=int, default=2)
    parser.add_argument("--rnn_size", dest="rnn_size", type=int, default=512)
    parser.add_argument("--max_word_length", dest="max_word_length", type=int, default=13)
    parser.add_argument("--filter_nums", dest="filter_nums", type=list, default=[32,32,32,64,128,128,128])
    parser.add_argument("--hightway", dest="highway", type=bool, default=True)
    parser.add_argument("--kernels", dest="kernels", type=list, default=[2,3,4,5,6,7,8])
    parser.add_argument("--index_dir", dest="index_dir", type=str, default="../data/char_index.txt")
    parser.add_argument("--itrs", dest="itrs", type=int, default=10001)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=3)
    parser.add_argument("--embedding_size", dest="embedding_size", default=128)
    parser.add_argument("--max_time_step", dest="max_time_step", type=int, default=25)
    parser.add_argument("--vocab_size", dest="vocab_size", type=int, default=3249)
    parser.add_argument("--train", dest="train", type=bool, default=True)
    parser.add_argument("--saved", dest="saved", type=str, default="save/")
    parser.add_argument("--keep_prob", dest="keep_prob", type=float, default=0.5)
    parser.add_argument("--test", dest='test', type=bool, default=True)
    args= parser.parse_args()

    if not os.path.exists(args.saved):
        os.mkdir(args.saved)
    
    if not os.path.exists(args.data_dir):
        mk_train_and_test_data(args.data_dir)

    model_ = model(args)
    if args.train:
        model_.train()
    
