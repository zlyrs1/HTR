conda activate HTR

cd ./HTR
echo $PWD

nohup python -u ./airsim_plugin/AirVLNSimulatorServerTool.py --gpus 0,1,2,3,4,5,6,7 &

python -u ./src/vlnce_src/train.py \
--run_type train \
--policy_type cma \
--collect_type TF \
--name HTR-cma \
--batchSize 8 \
--dagger_it 1 \
--epochs 100 \
--lr 0.00025 \
--trainer_gpu_device 0


python -u ./src/vlnce_src/dagger_train.py \
--run_type train \
--policy_type cma \
--collect_type dagger \
--name HTR-cma-dagger \
--batchSize 8 \
--dagger_it 10 \
--epochs 5 \
--lr 0.00025 \
--trainer_gpu_device 0 \
--dagger_update_size 5000

