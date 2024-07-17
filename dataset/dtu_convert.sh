##### Step 1: Scannetpp Initial
# cd scannetpp-tools
# python -m dslr.undistort_colmap dslr/configs/undistort_colmap.yml
# cd ..
##### Remove the unnecessary picture in scannetpp_pinhole




##### Step 2: Colmap Preprocess, generate npz and soon
# Convert all case & Generate 480*360 Images 480 360
# cd scannetpp-tools
# cd scannetpp_colmap_preprocess
# case_list=("6ee2fc1070" "7b6477cb95" "9460c8889d")
# for case in "${case_list[@]}"; do
#     python imgs2poses.py ../../raw/Scannetpp_pinhole/$case
#     python gen_cameras.py ../../raw/Scannetpp_pinhole/$case ../../processored/Scannetpp_crop_scaled/$case 480 360
# done
# cd ..
# cd ..
#####


##### Step 3: Normal Prediction
cd surface_normal_uncertainty
case_list=("scan24" "scan37" "scan40")
for case in "${case_list[@]}"; do
    python test.py --imgs_dir ../processored/DTU/$case/image --output_dir ../processored/DTU/$case/pred_normal
done
cd ..
##### 


##### Step 4: Dino Feature Extraction
case_list=("scan24" "scan37" "scan40")
for case in "${case_list[@]}"; do
    python ./dino_feature_extract/extractor.py\
        --image_path ./processored/DTU/$case/image \
        --output_path ./processored/DTU/$case
done
#####


##### Step 5: SAM Segmentation Extraction Typo
# case_list=("6ee2fc1070" "7b6477cb95" "9460c8889d")
# split_list=("split_sub20.pkl" "split_sub5.pkl" "split_sub10.pkl")
# length=${#case_list[@]}
# for ((i=0; i < $length; i++)); do
#     case=${case_list[$i]}
#     split=${split_list[$i]}
#     python ./sam_extract/extractor.py\
#         --image_path ./processored/Scannetpp_crop_scaled/$case/image \
#         --output_path ./processored/Scannetpp_crop_scaled/$case/sam \
#         --split_file ./processored/Scannetpp_crop_scaled/$case/$split

# done
#####
