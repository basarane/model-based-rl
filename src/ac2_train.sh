rm ../runs/run_ac2_test3 -rf
mkdir ../runs/run_ac2_test3/
python -B dqn.py Breakout --gpu 1 --output_dir ../runs/run_ac2_test3/ --actor-critic True --REPLAY_START_SIZE 32 # > ../runs/run_ac2_test3/log.txt & #
