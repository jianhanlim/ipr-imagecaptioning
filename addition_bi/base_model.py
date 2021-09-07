import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import cPickle as pickle
import copy
import json
from tqdm import tqdm

from utils.nn import NN
from utils.coco.coco import COCO
from utils.coco.pycocoevalcap.eval import COCOEvalCap
from utils.misc import ImageLoader, CaptionData, TopN, generate_binary_key
from datetime import datetime as dt
from concurrent.futures import ThreadPoolExecutor
import logging

from tensorflow.contrib.model_pruning.python import pruning


class BaseModel(object):
    def __init__(self, config):
        self.config = config
        self.is_train = True if config.phase == 'train' else False
        self.train_cnn = self.is_train and config.train_cnn
        self.image_shape = [224, 224, 3]
        if self.config.cnn == 'inceptionv4':
            self.image_shape = [299, 299, 3]
        self.nn = NN(config)
        self.global_step = tf.Variable(0,name = 'global_step',trainable = False)
        self.saver_cnn = None
        self.saver = None
        self.build()
        self.saveExecutor = ThreadPoolExecutor(max_workers=1)

    def build(self):
        raise NotImplementedError()
    
    def prune(self, sess, train_data):
        """ Prune the model. """
        print("Pruning the model...")
        config = self.config
        
        # Get, Print, and Edit Pruning Hyperparameters
        pruning_hparams = pruning.get_pruning_hparams()
        print("Pruning Hyperparameters:", pruning_hparams)

        # Change hyperparameters to meet our needs
        pruning_hparams.begin_pruning_step = 0
        pruning_hparams.end_pruning_step = 250
        pruning_hparams.pruning_frequency = 1
        pruning_hparams.sparsity_function_end_step = 250
        pruning_hparams.target_sparsity = .5

        # Create a pruning object using the pruning specification, sparsity seems to have priority over the hparam
        p = pruning.Pruning(pruning_hparams, global_step=self.global_step, sparsity=.5)
        prune_op = p.conditional_mask_update_op()
        
        sess.run(tf.global_variables_initializer())
        sess.run(self.zero_ops)
        tf.get_default_graph().finalize()
        if not os.path.exists(config.summary_dir):
            os.mkdir(config.summary_dir)
        train_writer = tf.summary.FileWriter(config.summary_dir,
                                             sess.graph)

        # initialize a real key
        key = generate_binary_key(config.num_lstm_units, config.key, config.seed)
        signkey = generate_binary_key(config.num_lstm_units, config.signkey, -1)

        for _ in tqdm(list(range(config.num_epochs)), desc='epoch'):
            for _ in tqdm(list(range(train_data.num_batches)), desc='batch'):
                before_time = dt.now()
                batch = train_data.next_batch()
                after_time = dt.now()
                image_files, sentences, masks, images = batch

                feed_dict = {self.images: images,
                             self.sentences: sentences,
                             self.masks: masks,
                             self.key: key,
                             self.signkey: signkey}

                _, global_step, summary = sess.run([self.accum_ops, self.increment_global_step, self.summary], feed_dict=feed_dict)

                if (global_step + 1) % config.accumulate_grads == 0:
                    sess.run(prune_op)
                    sess.run(self.opt_op)
                    sess.run(self.zero_ops)
                    logging.debug("Weight sparsities:", sess.run(tf.contrib.model_pruning.get_weight_sparsity()))

                train_writer.add_summary(summary, global_step)

                if (global_step + 1) % config.save_period == 0:
                    self.save(sess)


                search_time = dt.now()
                logging.debug("Load Images Time: {}".format((after_time - before_time).total_seconds()))
                logging.debug("Search Time: {}".format((search_time - after_time).total_seconds()))

            train_data.reset()
            train_data.start_executor()

        self.save(sess)
        train_writer.close()
        print("Final sparsity by layer (should be 0)", sess.run(tf.contrib.model_pruning.get_weight_sparsity()))
        print("Pruning complete.")
    
    def train(self, sess, train_data):
        """ Train the model. """
        print("Training the model...")
        config = self.config

        if not os.path.exists(config.summary_dir):
            os.mkdir(config.summary_dir)
        train_writer = tf.summary.FileWriter(config.summary_dir,
                                             sess.graph)

        # initialize a real key
        key = generate_binary_key(config.num_lstm_units, config.key, config.seed)
        signkey = generate_binary_key(config.num_lstm_units, config.signkey, -1)

        for _ in tqdm(list(range(config.num_epochs)), desc='epoch'):
            for _ in tqdm(list(range(train_data.num_batches)), desc='batch'):
                before_time = dt.now()
                batch = train_data.next_batch()
                after_time = dt.now()
                image_files, sentences, masks, images = batch

                feed_dict = {self.images: images,
                             self.sentences: sentences,
                             self.masks: masks,
                             self.key: key,
                             self.signkey: signkey}

                _, global_step, summary = sess.run([self.accum_ops, self.increment_global_step, self.summary], feed_dict=feed_dict)

                if (global_step + 1) % config.accumulate_grads == 0:
                    sess.run(self.opt_op)
                    sess.run(self.zero_ops)

                train_writer.add_summary(summary, global_step)

                if (global_step + 1) % config.save_period == 0:
                    self.save(sess)

                search_time = dt.now()
                logging.debug("Load Images Time: {}".format((after_time - before_time).total_seconds()))
                logging.debug("Search Time: {}".format((search_time - after_time).total_seconds()))

            train_data.reset()
            train_data.start_executor()

        self.save(sess)
        train_writer.close()
        print("Training complete.")

    def eval(self, sess, eval_gt_coco, eval_data, vocabulary):
        """ Evaluate the model using the COCO val2014 data. """
        print("Evaluating the model ...")
        config = self.config

        results = []
        if not os.path.exists(config.eval_result_dir):
            os.mkdir(config.eval_result_dir)
        
        signkey_accuracy_final = 0
        # Generate the captions for the images
        idx = 0
        for k in tqdm(list(range(eval_data.num_batches)), desc='batch'):
            before_time = dt.now()
            batch = eval_data.next_batch()
            after_time = dt.now()
            caption_data, signkey_accuracy = self.beam_search(sess, batch, vocabulary)
            signkey_accuracy_final += signkey_accuracy
            
            search_time = dt.now()
            print("Load Images Time: {}".format((after_time - before_time).total_seconds()))
            print("Search Time: {}".format((search_time - after_time).total_seconds()))

            fake_cnt = 0 if k<eval_data.num_batches-1 \
                         else eval_data.fake_count
            for l in range(eval_data.batch_size-fake_cnt):
                word_idxs = caption_data[l][0].sentence
                score = caption_data[l][0].score
                caption = vocabulary.get_sentence(word_idxs)
                results.append({'image_id': eval_data.image_ids[idx],
                                'caption': caption})
                idx += 1

                # Save the result in an image file, if requested
                if config.save_eval_result_as_image:
                    image_file = batch[0][l]
                    image_name = image_file.split(os.sep)[-1]
                    image_name = os.path.splitext(image_name)[0]
                    img = plt.imread(image_file)
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(caption)
                    plt.savefig(os.path.join(config.eval_result_dir,
                                             image_name+'_result.jpg'))

        fp = open(config.eval_result_file, 'wb')
        json.dump(results, fp)
        fp.close()

        # Evaluate these captions
        eval_result_coco = eval_gt_coco.loadRes(config.eval_result_file)
        scorer = COCOEvalCap(eval_gt_coco, eval_result_coco)
        scorer.evaluate()
        print("Evaluation complete.")
        print("Sign Key Accuracy: ")
        print(signkey_accuracy_final/eval_data.num_batches)

    def test(self, sess, test_data, vocabulary):
        """ Test the model using any given images. """
        print("Testing the model ...")
        config = self.config

        if not os.path.exists(config.test_result_dir):
            os.mkdir(config.test_result_dir)

        captions = []
        scores = []

        # Generate the captions for the images
        for k in tqdm(list(range(test_data.num_batches)), desc='path'):
            batch = test_data.next_batch()
            caption_data, _ = self.beam_search(sess, batch, vocabulary)

            fake_cnt = 0 if k<test_data.num_batches-1 \
                         else test_data.fake_count
            for l in range(test_data.batch_size-fake_cnt):
                word_idxs = caption_data[l][0].sentence
                score = caption_data[l][0].score
                caption = vocabulary.get_sentence(word_idxs)
                captions.append(caption)
                scores.append(score)

                # Save the result in an image file
                # Save the result in an image file, if requested
                if config.save_eval_result_as_image:
                    image_file = batch[0][l]
                    image_name = image_file.split(os.sep)[-1]
                    image_name = os.path.splitext(image_name)[0]
                    img = plt.imread(image_file)
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(caption)
                    plt.savefig(os.path.join(config.test_result_dir,
                                             image_name+'_result.jpg'))

        # Save the captions to a file
        results = pd.DataFrame({'image_files':test_data.image_files,
                                'caption':captions,
                                'prob':scores})
        results.to_csv(config.test_result_file)
        print("Testing complete.")

    def beam_search(self, sess, batch, vocabulary):
        """Use beam search to generate the captions for a batch of images."""
        # Feed in the images to get the contexts and the initial LSTM states
        config = self.config
        image_files, images = batch
        #images = self.image_loader.load_images(image_files)

        # initialize a key
        key = generate_binary_key(config.num_lstm_units, config.key, config.seed)
        
        # attack key
        if config.attack_key > 0:
            different = int(config.num_lstm_units * config.attack_key)
            key[:different] = key[:different] * -1
        
        # attack key - prune
        if config.attack_prune > 0:
            different = int(config.num_lstm_units * config.attack_prune)
            key[:different] = 0
        
        signkey = generate_binary_key(config.num_lstm_units, config.signkey, -1)
        signkey_accuracy = 0
        
        contexts, initial_memory, initial_output = sess.run(
            [self.conv_feats, self.initial_memory, self.initial_output],
            feed_dict = {self.images: images})

        partial_caption_data = []
        complete_caption_data = []
        for k in range(config.batch_size):
            initial_beam = CaptionData(sentence = [],
                                       memory = initial_memory[k],
                                       output = initial_output[k],
                                       score = 1.0)
            partial_caption_data.append(TopN(config.beam_size))
            partial_caption_data[-1].push(initial_beam)
            complete_caption_data.append(TopN(config.beam_size))

        # Run beam search
        for idx in range(config.max_caption_length):
            partial_caption_data_lists = []
            for k in range(config.batch_size):
                data = partial_caption_data[k].extract()
                partial_caption_data_lists.append(data)
                partial_caption_data[k].reset()

            num_steps = 1 if idx == 0 else config.beam_size
            for b in range(num_steps):
                if idx == 0:
                    last_word = np.zeros((config.batch_size), np.int32)
                else:
                    last_word = np.array([pcl[b].sentence[-1]
                                        for pcl in partial_caption_data_lists],
                                        np.int32)

                last_memory = np.array([pcl[b].memory
                                        for pcl in partial_caption_data_lists],
                                        np.float32)
                last_output = np.array([pcl[b].output
                                        for pcl in partial_caption_data_lists],
                                        np.float32)

                memory, output, scores = sess.run(
                    [self.memory, self.output, self.probs],
                    feed_dict = {self.contexts: contexts,
                                 self.last_word: last_word,
                                 self.last_memory: last_memory,
                                 self.last_output: last_output,
                                 self.key: key})
                
                # attack sign
                if config.attack_sign > 0:
                    different = int(config.num_lstm_units * config.attack_sign)
                    output[:,config.num_lstm_units-different:] = output[:,config.num_lstm_units-different:] * -1
                
                if idx == 0 and b == 0:
                    signkey_accuracy = np.sum((np.sign(output) == signkey).astype(float)) \
                                        / (config.batch_size * config.num_lstm_units)
                
                # Find the beam_size most probable next words
                for k in range(config.batch_size):
                    caption_data = partial_caption_data_lists[k][b]
                    words_and_scores = list(enumerate(scores[k]))
                    words_and_scores.sort(key=lambda x: -x[1])
                    words_and_scores = words_and_scores[0:config.beam_size+1]

                    # Append each of these words to the current partial caption
                    for w, s in words_and_scores:
                        sentence = caption_data.sentence + [w]
                        score = caption_data.score * s
                        beam = CaptionData(sentence,
                                           memory[k],
                                           output[k],
                                           score)
                        if vocabulary.words[w] == '.':
                            complete_caption_data[k].push(beam)
                        else:
                            partial_caption_data[k].push(beam)

        results = []
        for k in range(config.batch_size):
            if complete_caption_data[k].size() == 0:
                complete_caption_data[k] = partial_caption_data[k]
            results.append(complete_caption_data[k].extract(sort=True))

        return results, signkey_accuracy

    def save(self, sess):
        config = self.config
        """ Save the model. """
        print(" Saving the model")
        if config.save_mode == 'numpy':
            before_time = dt.now()
            config = self.config
            data = {v.name: v.eval() for v in tf.global_variables()}
            save_path = os.path.join(config.save_dir, str(self.global_step.eval()))
            after_time = dt.now()

            np.save(save_path, data)

            search_time = dt.now()
            print("Load Variable: {}".format((after_time - before_time).total_seconds()))
            print("Save: {}".format((search_time - after_time).total_seconds()))
        else:
            before_time = dt.now()
            save_path = os.path.join(config.save_dir, 'model')
            self.saver.save(sess, save_path, global_step=self.global_step.eval())
            after_time = dt.now()
            print("Saver: {}".format((after_time - before_time).total_seconds()))

        info_file = open(os.path.join(config.save_dir, "config.pickle"), "wb")
        config_ = copy.copy(config)
        config_.global_step = self.global_step.eval()
        pickle.dump(config_, info_file)
        info_file.close()
        print("Model saved.")

    def load(self, sess, model_file=None):
        """ Load the model. """
        config = self.config

        if config.load_mode == 'numpy':
            if model_file is not None:
                save_path = model_file
            else:
                info_path = os.path.join(config.save_dir, "config.pickle")
                info_file = open(info_path, "rb")
                config = pickle.load(info_file)
                global_step = config.global_step
                info_file.close()
                save_path = os.path.join(config.save_dir,
                                         str(global_step)+".npy")

            print("Loading the model from %s..." %save_path)
            data_dict = np.load(save_path).item()
            count = 0
            for v in tqdm(tf.global_variables()):
                if v.name in data_dict.keys():
                    try:
                        sess.run(v.assign(data_dict[v.name]))
                        count += 1
                    except Exception as e:
                        print(e)
                        print(v.name)
            print("%d tensors loaded." %count)
        else:
            #self.saver.restore(sess, model_file)
            self.optimistic_restore(sess, model_file)
            print("Model restored.")

    def optimistic_restore(self, session, save_file):
        reader = tf.train.NewCheckpointReader(save_file)
        saved_shapes = reader.get_variable_to_shape_map()
        var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables() if var.name.split(':')[0] in saved_shapes])
        restore_vars = []
        name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
        with tf.variable_scope('', reuse=True):
            for var_name, saved_var_name in var_names:
                curr_var = name2var[saved_var_name]
                var_shape = curr_var.get_shape().as_list()
                if var_shape == saved_shapes[saved_var_name]:
                    restore_vars.append(curr_var)
            print("number of parameters: {}".format(len(restore_vars)))
        saver = tf.train.Saver(restore_vars)
        saver.restore(session, save_file)

    def load_cnn(self, session, data_path, ignore_missing=True):
        """ Load a pretrained CNN model. """
        print("Loading the CNN from %s..." %data_path)

        if self.config.cnn in ['vgg16', 'resnet50']:
            data_dict = np.load(data_path).item()
            count = 0
            for op_name in tqdm(data_dict):
                with tf.variable_scope(op_name, reuse = True):
                    for param_name, data in data_dict[op_name].iteritems():
                        try:
                            var = tf.get_variable(param_name)
                            session.run(var.assign(data))
                            count += 1
                        except ValueError:
                            pass
            print("%d tensors loaded." %count)

        elif self.config.cnn in ['resnet101', 'inceptionv4']:
            # Restore tf checkpoint variables from disk.
            self.saver_cnn.restore(session, data_path)
            print("Model restored.")

        else:
            print('Incorrect CNN is selected')
            exit(-1)
