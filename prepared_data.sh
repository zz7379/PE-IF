EXPERIMENT_NAME=toy_example
PATH_TO_VIDEO=toy_example.MOV
SKIP_FRAME_RATE=24
SCENE_TYPE=object  # {outdoor,indoor,object}
bash projects/neuralangelo/scripts/preprocess.sh ${EXPERIMENT_NAME} ${PATH_TO_VIDEO} ${SKIP_FRAME_RATE} ${SCENE_TYPE}