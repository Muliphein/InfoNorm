python ./scannet_preprocessor.py \
    --input_path ./raw/ScanTest/living_room \
    --output_path ./processored/ScanTest/living_room \
    --max_length 480

# python ./dino_feature_extract/extractor.py\
#     --image_path ./raw/ScanTest/living_room/image \
#     --output_path ./processored/ScanTest/living_room

python ./scannet_preprocessor.py \
    --input_path ./raw/Scannet/scene0009_01 \
    --output_path ./processored/Scannet/scene0009 \
    --max_length 480

# python ./dino_feature_extract/extractor.py\
#     --image_path ./raw/Scannet/scene0009_01/image \
#     --output_path ./processored/Scannet/scene0009

python ./scannet_preprocessor.py \
    --input_path ./raw/Scannet/scene0050_00 \
    --output_path ./processored/Scannet/scene0050 \
    --max_length 480

# python ./dino_feature_extract/extractor.py\
#     --image_path ./raw/Scannet/scene0050_00/image \
#     --output_path ./processored/Scannet/scene0050

python ./scannet_preprocessor.py \
    --input_path ./raw/Scannet/scene0084_00 \
    --output_path ./processored/Scannet/scene0084 \
    --max_length 480

# python ./dino_feature_extract/extractor.py\
#     --image_path ./raw/Scannet/scene0084_00/image \
#     --output_path ./processored/Scannet/scene0084
