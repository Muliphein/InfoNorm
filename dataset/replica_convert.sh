# # Convert NPZ
# python ./replica_to_npz.py \
#     --input_path ./raw/Replica/office_0/Sequence_1 \
#     --output_path ./processored/Replica/office0_s1 \
#     --gtply_path ./raw/Replica/office_0/mesh.ply \
#     --max_length 480

# python ./replica_to_npz.py \
#     --input_path ./raw/Replica/office_1/Sequence_1 \
#     --output_path ./processored/Replica/office1_s1 \
#     --gtply_path ./raw/Replica/office_1/mesh.ply \
#     --max_length 480

# python ./replica_to_npz.py \
#     --input_path ./raw/Replica/office_2/Sequence_1 \
#     --output_path ./processored/Replica/office2_s1 \
#     --gtply_path ./raw/Replica/office_2/mesh.ply \
#     --max_length 480
    
# python ./replica_to_npz.py \
#     --input_path ./raw/Replica/office_3/Sequence_1 \
#     --output_path ./processored/Replica/office3_s1 \
#     --gtply_path ./raw/Replica/office_3/mesh.ply \
#     --max_length 480

# python ./replica_to_npz.py \
#     --input_path ./raw/Replica/office_4/Sequence_1 \
#     --output_path ./processored/Replica/office4_s1 \
#     --gtply_path ./raw/Replica/office_4/mesh.ply \
#     --max_length 480

# python ./replica_to_npz.py \
#     --input_path ./raw/Replica/room_0/Sequence_1 \
#     --output_path ./processored/Replica/room0_s1 \
#     --gtply_path ./raw/Replica/room_0/mesh.ply \
#     --max_length 480

# python ./replica_to_npz.py \
#     --input_path ./raw/Replica/room_1/Sequence_1 \
#     --output_path ./processored/Replica/room1_s1 \
#     --gtply_path ./raw/Replica/room_1/mesh.ply \
#     --max_length 480

# python ./replica_to_npz.py \
#     --input_path ./raw/Replica/room_2/Sequence_1 \
#     --output_path ./processored/Replica/room2_s1 \
#     --gtply_path ./raw/Replica/room_2/mesh.ply \
#     --max_length 480

#Convert Dino Feature
python ./dino_feature_extract/extractor.py\
    --image_path ./processored/Replica/office0_s1/image \
    --output_path ./processored/Replica/office0_s1 \
    --split_file ./processored/Replica/office0_s1/split_sub15.pkl

python ./dino_feature_extract/extractor.py\
    --image_path ./processored/Replica/office1_s1/image \
    --output_path ./processored/Replica/office1_s1 \
    --split_file ./processored/Replica/office1_s1/split_sub15.pkl

# python ./dino_feature_extract/extractor.py\
#     --image_path ./processored/Replica/office2_s1/image \
#     --output_path ./processored/Replica/office2_s1 \
#     --split_file ./processored/Replica/office2_s1/split_sub15.pkl

# python ./dino_feature_extract/extractor.py\
#     --image_path ./processored/Replica/office3_s1/image \
#     --output_path ./processored/Replica/office3_s1 \
#     --split_file ./processored/Replica/office3_s1/split_sub15.pkl

# python ./dino_feature_extract/extractor.py\
#     --image_path ./processored/Replica/office4_s1/image \
#     --output_path ./processored/Replica/office4_s1 \
#     --split_file ./processored/Replica/office4_s1/split_sub15.pkl

python ./dino_feature_extract/extractor.py\
    --image_path ./processored/Replica/room0_s1/image \
    --output_path ./processored/Replica/room0_s1 \
    --split_file ./processored/Replica/room0_s1/split_sub15.pkl

python ./dino_feature_extract/extractor.py\
    --image_path ./processored/Replica/room1_s1/image \
    --output_path ./processored/Replica/room1_s1 \
    --split_file ./processored/Replica/room1_s1/split_sub15.pkl

# python ./dino_feature_extract/extractor.py\
#     --image_path ./processored/Replica/room2_s1/image \
#     --output_path ./processored/Replica/room2_s1 \
#     --split_file ./processored/Replica/room2_s1/split_sub15.pkl

