rm ../runs/run_ac2_test1 -rf
mkdir ../runs/run_ac2_test1/
python -B dqn.py Breakout --output_dir ../runs/run_ac2_test1/ --actor-critic True  > ../runs/run_ac2_test1/log.txt & #--REPLAY_START_SIZE 2000 
