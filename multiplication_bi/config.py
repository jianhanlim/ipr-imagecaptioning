
class Config(object):
    """ Wrapper class for various (hyper)parameters. """
    def __init__(self):
        # about the model architecture
        self.cnn = 'resnet50'               # 'vgg16' or 'resnet50'
        self.max_caption_length = 20
        self.dim_embedding = 512
        self.num_lstm_units = 512
        self.num_initalize_layers = 2    # 1 or 2
        self.dim_initalize_layer = 512
        self.num_attend_layers = 2       # 1 or 2
        self.dim_attend_layer = 512
        self.num_decode_layers = 2       # 1 or 2
        self.dim_decode_layer = 1024
        self.key = 'this is a random key if no key is provided'                   # set the key
        self.seed = -1
        self.signkey = 'my signature for this image captioning model'              
        self.sign_alpha = 0.1

        # about the weight initialization and regularization
        self.fc_kernel_initializer_scale = 0.08
        self.fc_kernel_regularizer_scale = 1e-4
        self.fc_activity_regularizer_scale = 0.0
        self.conv_kernel_regularizer_scale = 1e-4
        self.conv_activity_regularizer_scale = 0.0
        self.fc_drop_rate = 0.5
        self.lstm_drop_rate = 0.3
        self.attention_loss_factor = 0.01
        # about the optimization
        self.num_epochs = 20
        self.batch_size = 32
        self.accumulate_grads = 1
        self.optimizer = 'Adam'    # 'Adam', 'RMSProp', 'Momentum' or 'SGD'
        self.initial_learning_rate = 0.0001 #0.0001 #0.00001
        self.learning_rate_decay_factor = 1.0 #0.95 #1.0
        self.num_steps_per_decay = 10000 #100000
        self.clip_gradients = 5.0
        self.momentum = 0.0
        self.use_nesterov = True
        self.decay = 0.9
        self.centered = True
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-6

        # about the saver
        self.save_period = 5000 #2000
        self.save_dir = './models/resnet50_ori'
        self.summary_dir = './summary/resnet50_ori'

        # about the vocabulary
        self.vocabulary_file = './vocabulary.csv'    # change different vocabulary size
        self.vocabulary_size = 10000 #8791 #5000    # change different vocabulary size

        # about the training
        self.train_image_dir = './train/images/'
        self.train_caption_file = './train/karpathy_train.json'
        #self.train_caption_file = './train/captions_train2014.json'
        self.temp_annotation_file = './train/anns.csv'      # change different vocabulary size
        self.temp_data_file = './train/data.npy'            # change different vocabulary size

        # about the evaluation
        self.eval_image_dir = './val/images/'
        self.eval_caption_file = './val/karpathy_test.json' #'./val/karpathy_val.json'
        #self.eval_caption_file = './val/captions_val2014.json'
        self.eval_result_dir = './val/results/'
        self.eval_result_file = './val/results.json'
        self.save_eval_result_as_image = False
        self.data_limit = -1

        # about the testing
        self.test_image_dir = './test/images/'
        self.test_result_dir = './test/results/'
        self.test_result_file = './test/results.csv'

        # about the multithreading
        self.num_thread = 2
        self.queue_size = self.num_thread * 2

    def update_dataset_setting(self):
        dt = self.dataset
        if dt == 'flickr30k':
            self.vocabulary_size = 10000 #8000    # change different vocabulary size
            self.vocabulary_file = './vocabulary_flickr30k_10k.csv' #'./vocabulary_flickr30k.csv'    # change different vocabulary size
            self.train_image_dir = './train/images_flickr30k/'
            self.train_caption_file = './train/karpathy_train_flickr30k.json'
            self.temp_annotation_file = './train/anns_flickr30k_10k.csv' #'./train/anns_flickr30k.csv'      # change different vocabulary size
            self.temp_data_file = './train/data_flickr30k_10k.npy' #'./train/data_flickr30k.npy'            # change different vocabulary size
            self.eval_image_dir = self.train_image_dir
            self.eval_caption_file = './val/karpathy_test_flickr30k.json' #'./val/karpathy_val.json'
            self.eval_result_dir = './val/results_flickr30k/'
            self.eval_result_file = './val/results_flickr30k.json'
        
        if dt == 'flickr8k':
            self.vocabulary_size = 5000 #8000    # change different vocabulary size
            self.vocabulary_file = './vocabulary_flickr8k_5k.csv' #'./vocabulary_flickr30k.csv'    # change different vocabulary size
            self.train_image_dir = './train/images_flickr8k/'
            self.train_caption_file = './train/karpathy_train_flickr8k.json'
            self.temp_annotation_file = './train/anns_flickr8k_5k.csv' #'./train/anns_flickr30k.csv'      # change different vocabulary size
            self.temp_data_file = './train/data_flickr8k_5k.npy' #'./train/data_flickr30k.npy'            # change different vocabulary size
            self.eval_image_dir = self.train_image_dir
            self.eval_caption_file = './val/karpathy_test_flickr8k.json' #'./val/karpathy_val.json'
            self.eval_result_dir = './val/results_flickr8k/'
            self.eval_result_file = './val/results_flickr8k.json'

    def __str__(self):
        return '%s(%s)' % (
            type(self).__name__,
            ', '.join('%s=%s' % item for item in vars(self).items())
        )
