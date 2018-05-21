rm test_line_dqn/linedqn-2 -rf
python -B oo_dqn.py Line --model LineModel --learning-rate 0.0003 --target-network-update 30 --egreedy-final 0.01 --egreedy-final-step 25000 --egreedy-decay 0.998 --replay-start-size 64 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --max-step 10000 --save-interval 1000 --logdir test_line_dqn/linedqn-2

# first line test
#mkdir test_line_dqn
#rm test_line_dqn/linedqn-1 -rf
#python -B oo_dqn.py Line --model LineModel --learning-rate 0.0003 --target-network-update 30 --egreedy-final 0.01 --egreedy-decay 0.99 --replay-start-size 64 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --max-step 5000 --save-interval 100 --logdir test_line_dqn/linedqn-1
