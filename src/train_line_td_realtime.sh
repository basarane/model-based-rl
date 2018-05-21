#ilk test
mkdir test_line_td_realtime
rm test_line_td_realtime/test-1 -rf
python -B oo_td_realtime.py Line --env-weightfile test_line_env/lineenv-1/weights_5000.h5 --batch-size 64 --max-step 50000 --learning-rate 0.001 --target-network-update 500000 --env-model EnvModelLine --vmodel LineVNetwork --save-freq 1000 --replay-start-size 100 --replay-buffer-size 10000 --egreedy-final 0.01 --egreedy-decay 0.998 --logdir test_line_td_realtime/test-1
