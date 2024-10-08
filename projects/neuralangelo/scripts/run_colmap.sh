# -----------------------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# -----------------------------------------------------------------------------

# usage: run_colmap.sh <project_path>

mkdir -p ${1}/sparse
mkdir -p ${1}/dense

colmap feature_extractor \
    --database_path ${1}/database.db \
    --image_path ${1}/raw_images \
    --ImageReader.camera_model=RADIAL \
    --SiftExtraction.use_gpu=0 \
    --SiftExtraction.num_threads=32 \
    --ImageReader.single_camera=true # assuming single camera

colmap sequential_matcher \
    --database_path ${1}/database.db \
    --SiftMatching.use_gpu=0

colmap mapper \
    --database_path ${1}/database.db \
    --image_path ${1}/raw_images \
    --output_path ${1}/sparse

colmap image_undistorter \
    --image_path ${1}/raw_images \
    --input_path ${1}/sparse/0 \
    --output_path ${1}/dense \
    --output_type COLMAP \
    --max_image_size 2000

rm -rf ${1}/sparse

PATH_TO_IMAGES=${1}
SCENE_TYPE=object  # {outdoor,indoor,object}
EXPERIMENT_NAME=${2}
python3 projects/neuralangelo/scripts/convert_data_to_json.py --data_dir ${PATH_TO_IMAGES}/dense --scene_type ${SCENE_TYPE}
python3 projects/neuralangelo/scripts/generate_config.py --experiment_name ${EXPERIMENT_NAME} --data_dir ${PATH_TO_IMAGES}/dense --scene_type ${SCENE_TYPE} --auto_exposure_wb
