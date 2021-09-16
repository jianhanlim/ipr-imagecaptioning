mkdir tmp

for i in $4
do
	# set config
	export CUDA_VISIBLE_DEVICES=$1
	MODEL_NAME=$2
	CNN='resnet50'
	BEAM_SIZE=3
	STEP=$i
	DATASET=$5 # 'flickr30k' #mscoco
	THREAD=1
	LOAD_MODE='tfsaver' # numpy or tfsaver
  DIM_EMBEDDING=512 # 512 or 1024
	LSTM_UNIT=512 # 512 or 1024
	KEY=$3
    SEED=$6

	# start evaluation
	python main.py --phase=eval \
	--dim_embedding=$DIM_EMBEDDING \
	--num_lstm_units=$LSTM_UNIT \
	--num_thread=$THREAD \
	--load_mode=$LOAD_MODE \
	--dataset=$DATASET \
	--cnn=$CNN \
	--key="$KEY" \
	--seed="$SEED" \
	--model_file='./models/'$MODEL_NAME'/'$STEP'' \
	--eval_result_file='val/'$MODEL_NAME'_'$STEP'_'$BEAM_SIZE'_'$KEY'.json' \
	--beam_size=$BEAM_SIZE > 'tmp/'$MODEL_NAME'_'$STEP'_'$BEAM_SIZE'_'$KEY'.txt'
done
