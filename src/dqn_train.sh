#python gym_test.py --env-model env --env-weight run_acvp_12/weights_200000.h5 --env-mean-image run_acvp_12/mean_image.npy --env-reward-weight run_acvp_12_reward/weights_200000.h5 --gpu 1 --REPLAY_MEMORY_SIZE 100000 Breakout run_dqn_env_3 > run_dqn_env_3/log.txt &
#mkdir run_dqn_rmsprop_4
#python gym_test.py --gpu 1 Breakout run_dqn_rmsprop_4 > run_dqn_rmsprop_4/log.txt &

#rm ../runs/run_breakout_dqn_1 -rf
#mkdir ../runs/run_breakout_dqn_1
#python -B dqn.py --gpu 1 Breakout --output_dir ../runs/run_breakout_dqn_1 > ../runs/run_breakout_dqn_1/log.txt &

#rm ../runs/run_breakout_ddqn_1 -rf
#mkdir ../runs/run_breakout_ddqn_1
#python -B dqn.py --gpu 1 Breakout --output_dir ../runs/run_breakout_ddqn_1 --double-dqn True > ../runs/run_breakout_ddqn_1/log.txt &

#rm ../runs/run_breakout_dueling_dqn_1 -rf
#mkdir ../runs/run_breakout_dueling_dqn_1
#python -B dqn.py --gpu 1 Breakout --output_dir ../runs/run_breakout_dueling_dqn_1 --dueling-network True > ../runs/run_breakout_dueling_dqn_1/log.txt &

#rm run_atlantis_dueling_dqn_1 -rf
#mkdir run_atlantis_dueling_dqn_1
#python gym_test.py Atlantis run_atlantis_dueling_dqn_1 --dueling-network True > run_atlantis_dueling_dqn_1/log.txt & # --enable-render True --render-step 1 #
#
#rm run_atlantis_dqn_1 -rf
#mkdir run_atlantis_dqn_1
#python gym_test.py Atlantis run_atlantis_dqn_1 > run_atlantis_dqn_1/log.txt & # --enable-render True --render-step 1 #

rm ../runs/run_breakout_dueling_ddqn_1 -rf
mkdir ../runs/run_breakout_dueling_ddqn_1
python -B dqn.py --gpu 1 Breakout --output_dir ../runs/run_breakout_dueling_ddqn_1 --dueling-network True --double-dqn True > ../runs/run_breakout_dueling_ddqn_1/log.txt &
