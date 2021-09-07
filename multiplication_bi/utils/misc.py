import numpy as np
import cv2
import heapq
import os
from datetime import datetime as dt
from concurrent.futures import ThreadPoolExecutor


class ImageLoader(object):
    def __init__(self, mean_file, shape, config):
        self.config = config
        self.bgr = True
        self.scale_shape = np.array([shape, shape], np.int32)
        self.crop_shape = np.array([shape, shape], np.int32)
        self.mean = np.load(mean_file).mean(1).mean(1)
        self.threadExecutor = ThreadPoolExecutor(max_workers=128)

    def preprocess(self, image):
        if self.config.cnn in ['vgg16','resnet50','resnet101']:
            image = cv2.resize(image, (self.scale_shape[0], self.scale_shape[1]))
            offset = (self.scale_shape - self.crop_shape) / 2
            offset = offset.astype(np.int32)
            image = image[offset[0]:offset[0] + self.crop_shape[0],
                    offset[1]:offset[1] + self.crop_shape[1]]
            image = image - self.mean
            return image
        elif self.config.cnn == 'inceptionv4':
            image = cv2.resize(image, (self.scale_shape[0], self.scale_shape[1]))
            image = np.array(image, np.float32)
            image /= 255.
            image -= 0.5
            image *= 2.

            return image
        else:
            return image

    def load_image(self, image_file):
        """ Load and preprocess an image. """
        image = cv2.imread(image_file)
        if self.bgr:
            # convert bgr to rgb
            temp = image.swapaxes(0, 2)
            temp = temp[::-1]
            image = temp.swapaxes(0, 2)

        return self.preprocess(image)

    def load_images(self, image_files):
        """ Load and preprocess a list of images. """
        #before_time = dt.now()
        images = []
        for image_file in image_files:
            images.append(self.load_image(image_file))
        # execs = []
        # for image_file in image_files:
        #     execs.append(self.threadExecutor.submit(self.load_image, image_file))
        # for exe in execs:
        #     images.append(exe.result())
        images = np.array(images, np.float32)
        #after_time = dt.now()
        #print("Load Images Time: {}".format((after_time-before_time).total_seconds()))
        return images


class CaptionData(object):
    def __init__(self, sentence, memory, output, score):
        self.sentence = sentence
        self.memory = memory
        self.output = output
        self.score = score

    def __cmp__(self, other):
        assert isinstance(other, CaptionData)
        if self.score == other.score:
            return 0
        elif self.score < other.score:
            return -1
        else:
            return 1

    def __lt__(self, other):
        assert isinstance(other, CaptionData)
        return self.score < other.score

    def __eq__(self, other):
        assert isinstance(other, CaptionData)
        return self.score == other.score


class TopN(object):
    def __init__(self, n):
        self._n = n
        self._data = []

    def size(self):
        assert self._data is not None
        return len(self._data)

    def push(self, x):
        assert self._data is not None
        if len(self._data) < self._n:
            heapq.heappush(self._data, x)
        else:
            heapq.heappushpop(self._data, x)

    def extract(self, sort=False):
        assert self._data is not None
        data = self._data
        self._data = None
        if sort:
            data.sort(reverse=True)
        return data

    def reset(self):
        self._data = []
        
def generate_binary_key(dims, sentence, seed):
    seed_key = np.ones(dims)
    if seed != -1:
        np.random.seed(seed)
        seed_key = np.random.randint(0, 2, dims)
        seed_key[seed_key<1] = -1 
    max_word = dims/8 
    new_sentence = sentence
    if len(sentence) < max_word: 
        left = max_word - len(sentence) 
        while left > 0: 
            index = left if left <= len(sentence) else len(sentence) 
            left -= len(sentence) 
            new_sentence = new_sentence + sentence[:int(index)] 
    new_sentence = new_sentence[:int(max_word)]
 
    binary_key = ''.join(format(ord(x), '08b') for x in new_sentence) 
    binary_key = ' '.join(list(binary_key)) 
    binary_key = np.fromstring(binary_key, dtype=int, sep=' ') 
    binary_key[binary_key<1] = -1
    binary_key = binary_key * seed_key
    return binary_key

def bits2str(key, seed, dims):
    seed_key = np.ones(dims)
    if seed != -1:
        np.random.seed(seed)
        seed_key = np.random.randint(0, 2, dims)
        seed_key[seed_key<1] = -1 
        
    key = key / seed_key
    key[key<1] = 0 
    key = key.astype(int)
    key = ''.join([item for item in key.astype(str)]) 
    return ''.join(chr(int(''.join(x), 2)) for x in zip(*[iter(key)]*8))

    
    
