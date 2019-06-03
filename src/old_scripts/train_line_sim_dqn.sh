# reward'lari da envmodel'den al
rm test_line_dqn_sim/linesim-1 -rf
python -B oo_dqn.py Line --model LineModel --learning-rate 0.0003 --target-network-update 30 --egreedy-final 0.01 --egreedy-decay 0.99 --replay-start-size 64 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --max-step 5000 --env-model EnvModelLine --env-weightfile test_line_env/lineenv-1/weights_5000.h5 --logdir test_line_dqn_sim/linesim-1
#--env-reward False 

