#GeoNeuS
cd code/geoneus
conf_list=("geoneus" "geoneus_jaco")
case_list=("Scannetpp_crop_scaled/0a7cc12c0e")
split_list=("split_sub5.pkl")
length=${#case_list[@]}
for conf in "${conf_list[@]}"; do
    for ((i=0; i < $length; i++)); do
        case=${case_list[$i]}
        split=${split_list[$i]}
        python exp_runner.py --mode train --conf ./confs/$conf.conf --case $case --split $split
    done
done

#i2sdf
cd code/i2-sdf
conf_list=("i2-sdf" "i2-sdf+jaco")
case_list=("Scannetpp_crop_scaled/0a7cc12c0e")
split_list=("split_sub5.pkl")
length=${#case_list[@]}
for conf in "${conf_list[@]}"; do
    for ((i=0; i < $length; i++)); do
        case=${case_list[$i]}
        split=${split_list[$i]}
        python data/normalize_cameras.py -i "../../dataset/processored/${case}/cameras_sphere.npz" -o "../../dataset/processored/${case}/cameras_normalize.npz"
        python main_recon.py --conf config/$conf.yml --scan_name $case --split $split --expname "${conf}/${split}/"
        python main_recon.py --conf config/$conf.yml --scan_name $case --test --test_mode mesh --split $split --expname "${conf}/${split}/"
    done
done

#MonoSDF
cd code/monosdf/code
conf_list=("scannet_mlp" "scannet_mlp_jaco")
case_list=("Scannetpp_crop_scaled/0a7cc12c0e")
split_list=("split_sub5.pkl")
length=${#case_list[@]}
for conf in "${conf_list[@]}"; do
    for ((i=0; i < $length; i++)); do
        case=${case_list[$i]}
        split=${split_list[$i]}
        python training/exp_runner.py --conf confs/$conf.conf  --scan_id $case --split $split
    done
done

#NeuralAngelo
cd code/NeuralAngelo
case_list=("0a7cc12c0e")
split_list=("split_sub5.pkl")
length=${#case_list[@]}
for ((i=0; i < $length; i++)); do

    case=${case_list[$i]}
    split=${split_list[$i]}
    echo "Run $case, $split"
    
    EXPERIMENT=$case
    GROUP=na
    NAME="${case}_${split}"
    CONFIG=projects/neuralangelo/configs/${EXPERIMENT}.yaml
    GPUS=1
    python train.py \
        --logdir=logs/${GROUP}/${NAME} \
        --config=${CONFIG} \
        --single_gpu \
        --split $split \
        # --wandb


    CONTENT=$(cat logs/${GROUP}/${NAME}/latest_checkpoint.txt)
    CHECKPOINT=logs/${GROUP}/${NAME}/${CONTENT}
    OUTPUT_MESH=logs/${GROUP}/${NAME}/150000.ply
    CONFIG=logs/${GROUP}/${NAME}/config.yaml
    RESOLUTION=512
    BLOCK_RES=128
    GPUS=1
    python projects/neuralangelo/scripts/extract_mesh.py \
        --config=${CONFIG} \
        --checkpoint=${CHECKPOINT} \
        --output_file=${OUTPUT_MESH} \
        --resolution=${RESOLUTION} \
        --block_res=${BLOCK_RES} \
        --single_gpu

    case="${case}_jaco"
    split=${split_list[$i]}
    echo "Run $case, $split"

    EXPERIMENT=$case
    GROUP=na
    NAME="${case}_${split}"
    CONFIG=projects/neuralangelo/configs/${EXPERIMENT}.yaml
    GPUS=1
    python train.py \
        --logdir=logs/${GROUP}/${NAME} \
        --config=${CONFIG} \
        --single_gpu \
        --split $split \
        # --wandb


    CONTENT=$(cat logs/${GROUP}/${NAME}/latest_checkpoint.txt)
    CHECKPOINT=logs/${GROUP}/${NAME}/${CONTENT}
    OUTPUT_MESH=logs/${GROUP}/${NAME}/150000.ply
    CONFIG=logs/${GROUP}/${NAME}/config.yaml
    RESOLUTION=512
    BLOCK_RES=128
    GPUS=1
    python projects/neuralangelo/scripts/extract_mesh.py \
        --config=${CONFIG} \
        --checkpoint=${CHECKPOINT} \
        --output_file=${OUTPUT_MESH} \
        --resolution=${RESOLUTION} \
        --block_res=${BLOCK_RES} \
        --single_gpu

done

#Neuris
cd code/monosdf/code
conf_list=("neuris" "neuris_jaco")
case_list=("Scannetpp_crop_scaled/0a7cc12c0e")
split_list=("split_sub5.pkl")
length=${#case_list[@]}
for conf in "${conf_list[@]}"; do
    for ((i=0; i < $length; i++)); do
        case=${case_list[$i]}
        split=${split_list[$i]}
        python exp_runner.py --mode train --conf ./confs/$conf.conf --case $case --split $split 
    done
done

#NeuS, It's involved by neuris
cd code/monosdf/code
conf_list=("neus" "neus_jaco")
case_list=("Scannetpp_crop_scaled/0a7cc12c0e")
split_list=("split_sub5.pkl")
length=${#case_list[@]}
for conf in "${conf_list[@]}"; do
    for ((i=0; i < $length; i++)); do
        case=${case_list[$i]}
        split=${split_list[$i]}
        python exp_runner.py --mode train --conf ./confs/$conf.conf --case $case --split $split 
    done
done


#Volsdf
cd code/volsdf/code
conf_list=("volsdf" "volsdf_jaco")
case_list=("Scannetpp_crop_scaled/0a7cc12c0e")
split_list=("split_sub5.pkl")
length=${#case_list[@]}
for conf in "${conf_list[@]}"; do
    for ((i=0; i < $length; i++)); do
        case=${case_list[$i]}
        split=${split_list[$i]}
        python training/exp_runner.py --conf ./confs/$conf.conf --scan_id $case --split $split --expname "${conf}/${split}/"
    done
done

