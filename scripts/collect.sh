
conda activate HTR

cd ./HTR
echo $PWD


nohup python -u ./airsim_plugin/AirVLNSimulatorServerTool.py --gpus 0,1,2,3,4,5,6,7 &

python -u ./src/vlnce_src/train.py \
--run_type collect \
--policy_type cma \
--collect_type TF \
--name HTR-cma \
--batchSize 16

