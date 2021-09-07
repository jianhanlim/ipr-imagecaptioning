import os
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

from datetime import datetime as dt
from utils.coco.coco import COCO
from utils.vocabulary import Vocabulary
from utils.misc import ImageLoader
from concurrent.futures import ThreadPoolExecutor
from Queue import Queue
import logging
import traceback
import sys
import time


class DataSet(object):
    def __init__(self,
                 image_ids,
                 image_files,
                 batch_size,
                 config,
                 word_idxs=None,
                 masks=None,
                 is_train=False,
                 shuffle=False):
        self.image_ids = np.array(image_ids)
        self.image_files = np.array(image_files)
        self.word_idxs = np.array(word_idxs)
        self.masks = np.array(masks)
        self.batch_size = batch_size
        self.is_train = is_train
        self.shuffle = shuffle
        self.config = config
        self.image_shape = [224, 224, 3]
        if self.config.cnn == 'inceptionv4':
            self.image_shape = [299, 299, 3]
        self.image_loader = ImageLoader('./utils/ilsvrc_2012_mean.npy', self.image_shape[0], config)
        self.executor = ThreadPoolExecutor(max_workers=config.num_thread)
        self.queue = Queue(maxsize=config.queue_size)
        self.lockQueue = Queue(maxsize=1) # use to control multithreading reading
        self.start = True                   # use to control when to shutdown all threads
        self.setup()

    def setup(self):
        """ Setup the dataset. """
        self.count = len(self.image_ids)
        self.num_batches = int(np.ceil(self.count * 1.0 / self.batch_size))
        self.fake_count = self.num_batches * self.batch_size - self.count
        self.idxs = list(range(self.count))
        self.reset()

    def stop(self):
        self.start = False

    def reset(self):
        """ Reset the dataset. """
        self.start = False
        time.sleep(4)

        with self.queue.mutex:
            self.queue.queue.clear()
        with self.lockQueue.mutex:
            self.lockQueue.queue.clear()

        self.current_idx = 0
        if self.shuffle:
            np.random.shuffle(self.idxs)

        self.start = True

    def next_batch(self):
        return self.queue.get()

    def start_executor(self):
        while self.start:
            try:
                self.lockQueue.put(True, timeout=3)
                break
            except:
                pass

        for x in range(self.config.num_thread):
            self.executor.submit(self.start_read_images, x+1)

    def start_read_images(self, x):
        while self.start and self.has_next_batch():
            try:
                while self.lockQueue.get(timeout=3) and self.has_next_batch():
                    logging.debug("Queue Size {}: {}".format(x, self.queue.qsize()))
                    next_batch = self.read_next_batch()
                    while self.start:
                        try:
                            self.queue.put(next_batch, timeout=3)
                            break
                        except:
                            pass
            except:
                pass

    def read_next_batch(self):
        """ Fetch the next batch. """
        assert self.has_next_batch()

        if self.has_full_next_batch():
            start, end = self.current_idx, \
                         self.current_idx + self.batch_size
            current_idxs = self.idxs[start:end]
        else:
            start, end = self.current_idx, self.count
            current_idxs = self.idxs[start:end] + \
                           list(np.random.choice(self.count, self.fake_count))

        logging.debug("Start: {}, End: {}".format(start, end))
        self.current_idx += self.batch_size
        if len(current_idxs) > self.batch_size:         # double confirm it won't exceed batch size
            current_idxs = current_idxs[:self.batch_size]

        while self.start:
            try:
                self.lockQueue.put(True, timeout=3) # multithreading locking
                break
            except:
                pass

        image_files = self.image_files[current_idxs]
        before_time = dt.now()
        images = self.image_loader.load_images(image_files)
        after_time = dt.now()
        logging.debug("Load Images: {}".format((after_time-before_time).total_seconds()))

        if self.is_train:
            word_idxs = self.word_idxs[current_idxs]
            masks = self.masks[current_idxs]
            #self.current_idx += self.batch_size
            return image_files, word_idxs, masks, images
        else:
            #self.current_idx += self.batch_size
            return image_files, images

    def has_next_batch(self):
        """ Determine whether there is a batch left. """
        return self.current_idx < self.count

    def has_full_next_batch(self):
        """ Determine whether there is a full batch left. """
        return self.current_idx + self.batch_size <= self.count

def prepare_train_data(config, sess):
    """ Prepare the data for training the model. """
    coco = COCO(config.train_caption_file)
    coco.filter_by_cap_len(config.max_caption_length)

    print("Building the vocabulary...")
    vocabulary = Vocabulary(config.vocabulary_size)
    if not os.path.exists(config.vocabulary_file):
        vocabulary.build(coco.all_captions())
        vocabulary.save(config.vocabulary_file)
    else:
        vocabulary.load(config.vocabulary_file)
    print("Vocabulary built.")
    print("Number of words = %d" %(vocabulary.size))

    coco.filter_by_words(set(vocabulary.words))

    print("Processing the captions...")
    if not os.path.exists(config.temp_annotation_file):
        captions = [coco.anns[ann_id]['caption'] for ann_id in coco.anns]
        image_ids = [coco.anns[ann_id]['image_id'] for ann_id in coco.anns]
        #image_files = [os.path.join(config.train_image_dir,
        #                            coco.imgs[image_id]['file_name'])
        #                            for image_id in image_ids]
        image_files = []
        for image_id in image_ids:
            filename = coco.imgs[image_id]['file_name']
            folderpath = config.train_image_dir
            if 'train2014' not in filename:
                folderpath = config.eval_image_dir
            i_path = os.path.join(folderpath, filename)
            image_files.append(i_path)

        annotations = pd.DataFrame({'image_id': image_ids,
                                    'image_file': image_files,
                                    'caption': captions})
        annotations.to_csv(config.temp_annotation_file)
    else:
        annotations = pd.read_csv(config.temp_annotation_file)
        captions = annotations['caption'].values
        image_ids = annotations['image_id'].values
        image_files = annotations['image_file'].values

    if not os.path.exists(config.temp_data_file):
        word_idxs = []
        masks = []
        for caption in tqdm(captions):
            current_word_idxs_ = vocabulary.process_sentence(caption)
            current_num_words = len(current_word_idxs_)
            current_word_idxs = np.zeros(config.max_caption_length,
                                         dtype = np.int32)
            current_masks = np.zeros(config.max_caption_length)
            current_word_idxs[:current_num_words] = np.array(current_word_idxs_)
            current_masks[:current_num_words] = 1.0
            word_idxs.append(current_word_idxs)
            masks.append(current_masks)
        word_idxs = np.array(word_idxs)
        masks = np.array(masks)
        data = {'word_idxs': word_idxs, 'masks': masks}
        np.save(config.temp_data_file, data)
    else:
        data = np.load(config.temp_data_file).item()
        word_idxs = data['word_idxs']
        masks = data['masks']
    print("Captions processed.")
    print("Number of captions = %d" %(len(captions)))

    print("Building the dataset...")
    dataset = DataSet(image_ids,
                      image_files,
                      config.batch_size,
                      config,
                      word_idxs,
                      masks,
                      True,
                      True)
    print("Dataset built.")
    return dataset

def prepare_eval_data(config, sess):
    """ Prepare the data for evaluating the model. """
    coco = COCO(config.eval_caption_file, data_limit=config.data_limit)
    image_ids = list(coco.imgs.keys())
    image_ids = image_ids
    image_files = [os.path.join(config.eval_image_dir,
                                coco.imgs[image_id]['file_name'])
                                for image_id in image_ids]

    print("Building the vocabulary...")
    if os.path.exists(config.vocabulary_file):
        vocabulary = Vocabulary(config.vocabulary_size,
                                config.vocabulary_file)
    else:
        vocabulary = build_vocabulary(config)
    print("Vocabulary built.")
    print("Number of words = %d" %(vocabulary.size))

    print("Building the dataset...")
    dataset = DataSet(image_ids, image_files, config.batch_size, config)
    print("Dataset built.")
    return coco, dataset, vocabulary

def prepare_test_data(config, sess):
    """ Prepare the data for testing the model. """
    files = os.listdir(config.test_image_dir)
    image_files = [os.path.join(config.test_image_dir, f) for f in files
        if f.lower().endswith('.jpg') or f.lower().endswith('.jpeg')]
    image_ids = list(range(len(image_files)))

    print("Building the vocabulary...")
    if os.path.exists(config.vocabulary_file):
        vocabulary = Vocabulary(config.vocabulary_size,
                                config.vocabulary_file)
    else:
        vocabulary = build_vocabulary(config)
    print("Vocabulary built.")
    print("Number of words = %d" %(vocabulary.size))

    print("Building the dataset...")
    dataset = DataSet(image_ids, image_files, config.batch_size, config)
    print("Dataset built.")
    return dataset, vocabulary

def build_vocabulary(config):
    """ Build the vocabulary from the training data and save it to a file. """
    coco = COCO(config.train_caption_file)
    coco.filter_by_cap_len(config.max_caption_length)

    vocabulary = Vocabulary(config.vocabulary_size)
    vocabulary.build(coco.all_captions())
    vocabulary.save(config.vocabulary_file)
    return vocabulary
