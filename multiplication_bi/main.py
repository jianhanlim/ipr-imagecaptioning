#!/usr/bin/python
import tensorflow as tf

from config import Config
from model import CaptionGenerator
from dataset import prepare_train_data, prepare_eval_data, prepare_test_data
import logging
import os, signal
logging.basicConfig(
    filename='train.log',
    filemode='w',
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')
logging.info("testing")
logging.debug("Start training")

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string('phase', 'train',
                       'The phase can be train, eval or test')

tf.flags.DEFINE_boolean('load', False,
                        'Turn on to load a pretrained model from either \
                        the latest checkpoint or a specified file')

tf.flags.DEFINE_string('model_file', None,
                       'If sepcified, load a pretrained model from this file')

tf.flags.DEFINE_boolean('load_cnn', False,
                        'Turn on to load a pretrained CNN model')

tf.flags.DEFINE_string('cnn_model_file', './vgg16_no_fc.npy',
                       'The file containing a pretrained CNN model')

tf.flags.DEFINE_boolean('train_cnn', False,
                        'Turn on to train both CNN and RNN. \
                         Otherwise, only RNN is trained')

tf.flags.DEFINE_integer('beam_size', 3,
                        'The size of beam search for caption generation')

# config set up
tf.flags.DEFINE_string('cnn', 'resnet50',
                       'vgg16 or resnet50')

tf.flags.DEFINE_string('globalpool', 'no',
                       'resnet101 global pooling, yes or no')

tf.flags.DEFINE_string('save_dir', 'no_name',
                       'The name to be saved for model and summary')

tf.flags.DEFINE_float('initial_learning_rate', 0.0001,
                       'the initial learning rate')

tf.flags.DEFINE_string('dataset', 'mscoco',
                       'mscoco or flickr30k')

tf.flags.DEFINE_integer('num_thread', 1,
                        'number of threads')

tf.flags.DEFINE_string('load_mode', 'numpy',
                       'numpy or tfsaver to load the model')

tf.flags.DEFINE_string('save_mode', 'numpy',
                       'numpy or tfsaver to save the model')

tf.flags.DEFINE_integer('save_period', 5000,
                        'Save every x iterations')

tf.flags.DEFINE_integer('dim_embedding', 512,
                        'number of embedding dimension')

tf.flags.DEFINE_integer('num_lstm_units', 512,
                        'number of lstm units')

tf.flags.DEFINE_string('eval_result_file', None,
                       'If sepcified, set the eval json filepath')

tf.flags.DEFINE_integer('num_epochs', 20,
                        'number of epochs')

tf.flags.DEFINE_string('key', 'this is a random key if no key is provided',
                        'key: maximum 64 ascii characters')
                        
tf.flags.DEFINE_integer('seed', -1,
                        'seed for key, -1 as no seed')
                        
tf.flags.DEFINE_string('signkey', 'my signature for this image captioning model',
                        'signkey: maximum 64 ascii characters')
                        
tf.flags.DEFINE_float('sign_alpha', 0.1,
                       'to avoid sign loss not trainable')  
                                       
tf.flags.DEFINE_float('attack_key', 0,
                       '0 to 1, 0 mean no disimilarity')
                       
tf.flags.DEFINE_float('attack_prune', 0,
                       '0 to 1, 0 mean no prune')
                       
tf.flags.DEFINE_float('attack_sign', 0,
                       '0 to 1, 0 mean no disimilarity')

def main(argv):
    config = Config()
    config.phase = FLAGS.phase
    config.train_cnn = FLAGS.train_cnn
    config.beam_size = FLAGS.beam_size
    config.cnn = FLAGS.cnn
    config.globalpool = FLAGS.globalpool
    config.save_dir = './models/' + FLAGS.save_dir
    config.summary_dir = './summary/summary_' + FLAGS.save_dir
    config.initial_learning_rate = FLAGS.initial_learning_rate
    config.num_thread = FLAGS.num_thread
    config.queue_size = config.num_thread * 2
    config.load_mode = FLAGS.load_mode
    config.save_mode = FLAGS.save_mode
    config.save_period = FLAGS.save_period
    config.dim_embedding = FLAGS.dim_embedding
    config.num_lstm_units = FLAGS.num_lstm_units
    config.num_epochs = FLAGS.num_epochs
    config.key = FLAGS.key
    config.seed = FLAGS.seed
    config.signkey = FLAGS.signkey
    config.sign_alpha = FLAGS.sign_alpha
    config.attack_key = FLAGS.attack_key
    config.attack_prune = FLAGS.attack_prune
    config.attack_sign = FLAGS.attack_sign
    config.dataset = FLAGS.dataset
    config.update_dataset_setting()
    if FLAGS.eval_result_file:
        config.eval_result_file = FLAGS.eval_result_file

    print('Checking config file...........')
    print(config.__str__())

    configT = tf.ConfigProto()
    configT.gpu_options.allow_growth = True
    with tf.Session(config=configT) as sess:
    #with tf.Session() as sess:
        if FLAGS.phase == 'train':
            # training phase
            data = prepare_train_data(config,sess)
            model = CaptionGenerator(config)
            sess.run(tf.global_variables_initializer())
            sess.run(model.zero_ops)
            if FLAGS.load:
                model.load(sess, FLAGS.model_file)
            if FLAGS.load_cnn:
                model.load_cnn(sess, FLAGS.cnn_model_file)
            tf.get_default_graph().finalize()
            data.start_executor()
            model.train(sess, data)
            data.stop()

        elif FLAGS.phase == 'eval':
            # evaluation phase
            coco, data, vocabulary = prepare_eval_data(config,sess)
            model = CaptionGenerator(config)
            model.load(sess, FLAGS.model_file)
            tf.get_default_graph().finalize()
            data.start_executor()
            model.eval(sess, coco, data, vocabulary)
            data.stop()
        else:
            # testing phase
            data, vocabulary = prepare_test_data(config,sess)
            model = CaptionGenerator(config)
            model.load(sess, FLAGS.model_file)
            tf.get_default_graph().finalize()
            data.start_executor()
            model.test(sess, data, vocabulary)
            data.stop()

if __name__ == '__main__':
    tf.app.run()
