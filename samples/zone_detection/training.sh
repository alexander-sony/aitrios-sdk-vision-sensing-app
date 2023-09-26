#! /usr/bin/bash
TRAINING_DOCKER_VOLUME_DIR=/root/samples/zone_detection
TRAINING_DOCKER_IMAGE_NAME=tf1_od_api_env:1.0.0
HOME=work
MODELS_DIR=$HOME/models

NUM_TRAIN_STEPS=300
PIPELINE_CONFIG_PATH=$TRAINING_DOCKER_VOLUME_DIR/$MODELS_DIR/pipeline.config
MODEL_DIR=$TRAINING_DOCKER_VOLUME_DIR/$MODELS_DIR/out/train
SAMPLE_1_OF_N_EVAL_EXAMPLES=100
docker run --rm -t -v $PWD:$TRAINING_DOCKER_VOLUME_DIR \
    --network host $TRAINING_DOCKER_IMAGE_NAME \
    python /tensorflow/models/research/object_detection/model_main.py \
    --pipeline_config_path=$PIPELINE_CONFIG_PATH \
    --model_dir=$MODEL_DIR \
    --num_train_steps=$NUM_TRAIN_STEPS \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr