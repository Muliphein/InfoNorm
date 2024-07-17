##### Step 1: Scannetpp Initial
# cd scannetpp-tools
# python -m dslr.undistort_colmap dslr/configs/undistort_colmap.yml
# cd ..
##### Remove the unnecessary picture in scannetpp_pinhole




##### Step 2: Colmap Preprocess, generate npz and soon
# Convert all case & Generate 480*360 Images 480 360
# cd scannetpp-tools
# cd scannetpp_colmap_preprocess
# case_list=("0a7cc12c0e")
# for case in "${case_list[@]}"; do
#     python imgs2poses.py ../../raw/Scannetpp_pinhole/$case
#     python gen_cameras.py ../../raw/Scannetpp_pinhole/$case ../../processored/Scannetpp_crop_scaled/$case 480 360
# done
# cd ..
# cd ..
#####


##### Step 3: Normal Prediction
# cd surface_normal_uncertainty
# case_list=("0a7cc12c0e")
# for case in "${case_list[@]}"; do
#     python test.py --imgs_dir ../processored/Scannetpp_crop_scaled/$case/image --output_dir ../processored/Scannetpp_crop_scaled/$case/pred_normal
# done
# cd ..
##### 


##### Step 4: Dino Feature Extraction
# case_list=("0a7cc12c0e")
# for case in "${case_list[@]}"; do
#     python ./dino_feature_extract/extractor.py\
#         --image_path ./processored/Scannetpp_crop_scaled/$case/image \
#         --output_path ./processored/Scannetpp_crop_scaled/$case
# done
#####