
conda activate HTR

cd ./HTR
echo $PWD


nohup python -u ./airsim_plugin/AirVLNSimulatorServerTool.py --gpus 0,1,2,3,4,5,6,7 &

python -u ./src/vlnce_src/train.py \
--run_type eval \
--policy_type cma \
--collect_type TF \
--name HTR-cma \
--batchSize 1 \
--EVAL_CKPT_PATH_DIR ../DATA/output/HTR-cma/train/checkpoint/ckpt.LAST.pth \
--EVAL_DATASET val_unseen \
--EVAL_NUM -1

