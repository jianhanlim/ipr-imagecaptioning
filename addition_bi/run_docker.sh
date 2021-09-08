# run tensorflow docker to train and eval
# For MSCOCO #############################################################
CODE_DIR=<absolute path to code repo>/ipr-imagecaptioning/addition_bi
CNN_DIR=<absolute path to pretrained cnn weight folder>/cnn-weight
DATA_DIR=<absolute path to mscoco dataset folder>/mscoco
COCO_EVAL_DIR=<absolute path to the pycocoevalcap>/pycocoevalcap

sudo docker run -it --rm --runtime=nvidia \
-v $CODE_DIR:/image_captioning_IPR \
-v $CNN_DIR:/image_captioning_IPR/cnn_models \
-v $DATA_DIR/train2014:/image_captioning_IPR/train/images \
-v $DATA_DIR/val2014:/image_captioning_IPR/val/images \
-v $DATA_DIR/annotations/karpathy_train.json:/image_captioning_IPR/train/karpathy_train.json \
-v $DATA_DIR/annotations/karpathy_test.json:/image_captioning_IPR/val/karpathy_test.json \
-v $COCO_EVAL_DIR:/image_captioning_IPR/utils/coco/pycocoevalcap \
-w /image_captioning_IPR \
limjianhan/tensorflow:1.13.1-gpu /bin/bash

# For Flickr30k ############################################################
CODE_DIR=<absolute path to code repo>/ipr-imagecaptioning/addition_bi
CNN_DIR=<absolute path to pretrained cnn weight folder>/cnn-weight
DATA_DIR=<absolute path to flickr30k dataset folder>/flickr30k
COCO_EVAL_DIR=<absolute path to the pycocoevalcap>/pycocoevalcap

sudo docker run -it --rm --runtime=nvidia \
-v $CODE_DIR:/image_captioning_IPR \
-v $CNN_DIR:/image_captioning_IPR/cnn_models \
-v $DATA_DIR/images_flickr30k:/image_captioning_IPR/train/images_flickr30k \
-v $DATA_DIR/annotations/karpathy_train_flickr30k.json:/image_captioning_IPR/train/karpathy_train_flickr30k.json \
-v $DATA_DIR/annotations/karpathy_test_flickr30k.json:/image_captioning_IPR/val/karpathy_test_flickr30k.json \
-v $COCO_EVAL_DIR:/image_captioning_IPR/utils/coco/pycocoevalcap \
-w /image_captioning_IPR \
limjianhan/tensorflow:1.13.1-gpu /bin/bash

# For Flickr8k ############################################################
CODE_DIR=<absolute path to code repo>/ipr-imagecaptioning/addition_bi
CNN_DIR=<absolute path to pretrained cnn weight folder>/cnn-weight
DATA_DIR=<absolute path to flickr8k dataset folder>/flickr8k
COCO_EVAL_DIR=<absolute path to the pycocoevalcap>/pycocoevalcap

sudo docker run -it --rm --runtime=nvidia \
-v $CODE_DIR:/image_captioning_IPR \
-v $CNN_DIR:/image_captioning_IPR/cnn_models \
-v $DATA_DIR/images_flickr8k:/image_captioning_IPR/train/images_flickr8k \
-v $DATA_DIR/annotations/karpathy_train_flickr8k.json:/image_captioning_IPR/train/karpathy_train_flickr8k.json \
-v $DATA_DIR/annotations/karpathy_test_flickr8k.json:/image_captioning_IPR/val/karpathy_test_flickr8k.json \
-v $COCO_EVAL_DIR:/image_captioning_IPR/utils/coco/pycocoevalcap \
-w /image_captioning_IPR \
limjianhan/tensorflow:1.13.1-gpu /bin/bash
