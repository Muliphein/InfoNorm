# cd omnidata-tools

# case_list=("0a7cc12c0e" "0a184cf634" "6ee2fc1070" "7b6477cb95" "56a0ec536c" "9460c8889d" "a08d9a2476" "e0abd740ba" "f8062cb7ce")

# for case in "${case_list[@]}"; do
#     python demo.py --task normal \
#         --img_path ../processored/Scannetpp_crop_scaled/$case/image \
#         --output_path ../processored/Scannetpp_crop_scaled/$case/omni_set

#     python demo.py --task depth \
#         --img_path ../processored/Scannetpp_crop_scaled/$case/image \
#         --output_path ../processored/Scannetpp_crop_scaled/$case/omni_set
# done

# cd ..

cd omnidata-tools

case_list=("scan24" "scan37" "scan40")

for case in "${case_list[@]}"; do
    python demo.py --task normal \
        --img_path ../processored/DTU/$case/image \
        --output_path ../processored/DTU/$case/omni_set

    python demo.py --task depth \
        --img_path ../processored/DTU/$case/image \
        --output_path ../processored/DTU/$case/omni_set
done

cd ..