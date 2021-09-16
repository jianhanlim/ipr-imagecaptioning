# set config
export CUDA_VISIBLE_DEVICES=$1
MODEL_NAME=$2
LEARNING_RATE=0.00001 #0.00001 #0.0001
CNN='resnet50' # vgg16 resnet50 resnet101 inceptionv4
DATASET=$5 #'flickr30k' #mscoco
THREAD=10
LOAD_MODE='tfsaver' # numpy or tfsaver
SAVE_MODE='tfsaver' # numpy or tfsaver
SAVE_PERIOD=5000
DIM_EMBEDDING=512 # 512 or 1024
LSTM_UNIT=512 # 512 or 1024
NUM_EPOCHS=12
KEY=$3
TRAINED_MODEL=$4
SEED=$7

# create summary and model folder
mkdir -p models/$MODEL_NAME
mkdir -p summary/'summary_'$MODEL_NAME

# start training
python main.py --phase=train \
--dim_embedding=$DIM_EMBEDDING \
--num_lstm_units=$LSTM_UNIT \
--num_thread=$THREAD \
--load_mode=$LOAD_MODE \
--save_mode=$SAVE_MODE \
--save_period=$SAVE_PERIOD \
--dataset=$DATASET \
--cnn=$CNN \
--initial_learning_rate=$LEARNING_RATE \
--save_dir=$MODEL_NAME \
--num_epochs=$NUM_EPOCHS \
--key="$KEY" \
--seed="$SEED" \
--load \
--model_file='./models/'$TRAINED_MODEL'/'$6 \
--train_cnn
