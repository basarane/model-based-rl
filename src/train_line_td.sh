rm test_line_td/linetd-24 -rf
python -B oo_td.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork  --save-freq 10 --smin -1 --smax 1 --sample-count 5000 --logdir test_line_td/linetd-24

# use sqr(td_error) as the probability of sampling, 5 tests in parallel
#rm test_line_td/linetd-23 -rf
#python -B oo_td.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork  --save-freq 10 --logdir test_line_td/linetd-23 &
#
#rm test_line_td/linetd-22 -rf
#python -B oo_td.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork  --save-freq 10 --logdir test_line_td/linetd-22 &
#
#rm test_line_td/linetd-21 -rf
#python -B oo_td.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork  --save-freq 10 --logdir test_line_td/linetd-21 &
#
#rm test_line_td/linetd-20 -rf
#python -B oo_td.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork  --save-freq 10 --logdir test_line_td/linetd-20 &
#
#rm test_line_td/linetd-19 -rf
#python -B oo_td.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork  --save-freq 10 --logdir test_line_td/linetd-19 &


# same run 5 times in parallel
#rm test_line_td/linetd-18 -rf
#python -B oo_td.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork  --save-freq 10 --logdir test_line_td/linetd-18 &
#
#rm test_line_td/linetd-17 -rf
#python -B oo_td.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork  --save-freq 10 --logdir test_line_td/linetd-17 &
#
#rm test_line_td/linetd-16 -rf
#python -B oo_td.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork  --save-freq 10 --logdir test_line_td/linetd-16 &
#
#rm test_line_td/linetd-15 -rf
#python -B oo_td.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork  --save-freq 10 --logdir test_line_td/linetd-15 &
#
#rm test_line_td/linetd-14 -rf
#python -B oo_td.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork  --save-freq 10 --logdir test_line_td/linetd-14 &

# sample states according to the abs of the td_error, if error is high, the probability of selection is high
#rm test_line_td/linetd-12 -rf
#python -B oo_td.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork  --save-freq 10 --logdir test_line_td/linetd-12

# same with previous
#rm test_line_td/linetd-11 -rf
#python -B oo_td.py Line --batch-size 64 --max-step 10000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork  --save-freq 10 --logdir test_line_td/linetd-11

# switch to truncated normal random state: (mean, std) = (0.5, 0.4)
#rm test_line_td/linetd-10 -rf
#python -B oo_td.py Line --batch-size 64 --max-step 10000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork  --save-freq 10 --logdir test_line_td/linetd-10

# same 5 more times in parallel
#rm test_line_td/linetd-9 -rf
#python -B oo_td.py Line --batch-size 64 --max-step 10000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork  --save-freq 10 --logdir test_line_td/linetd-9 &
#
#rm test_line_td/linetd-8 -rf
#python -B oo_td.py Line --batch-size 64 --max-step 10000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork  --save-freq 10 --logdir test_line_td/linetd-8 &
#
#rm test_line_td/linetd-7 -rf
#python -B oo_td.py Line --batch-size 64 --max-step 10000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork  --save-freq 10 --logdir test_line_td/linetd-7 &
#
#rm test_line_td/linetd-6 -rf
#python -B oo_td.py Line --batch-size 64 --max-step 10000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork  --save-freq 10 --logdir test_line_td/linetd-6 &
#
#rm test_line_td/linetd-5 -rf
#python -B oo_td.py Line --batch-size 64 --max-step 10000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork  --save-freq 10 --logdir test_line_td/linetd-5 &

# same with 10000 steps
#rm test_line_td/linetd-4 -rf
#python -B oo_td.py Line --batch-size 64 --max-step 10000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork  --save-freq 10 --logdir test_line_td/linetd-4

# td'deki sorun duzeltildi, td-realtime calisiyor, state [-1, 1] arasinda uniform random secildi, manual env model
#rm test_line_td/linetd-3 -rf
#python -B oo_td.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork  --save-freq 10 --logdir test_line_td/linetd-3

#rm test_line_td/linetd-2 -rf
#python -B oo_td.py Line --env-weightfile test_line_env/lineenv-2/weights_20000.h5 --batch-size 64 --max-step 1000 --learning-rate 0.001 --target-network-update 500000 --env-model EnvModelLine --vmodel LineVNetwork  --load-trajectory test_line_dqn/traj-2_mix.h5 --save-freq 1 --logdir test_line_td/linetd-2

#mkdir test_line_td
#rm test_line_td/linetd-1 -rf
#python -B oo_td.py Line --env-weightfile test_line_env/lineenv-1/weights_5000.h5 --batch-size 64 --max-step 1000 --learning-rate 0.001 --target-network-update 500000 --env-model EnvModelLine --vmodel LineVNetwork  --load-trajectory test_line_dqn/traj-1_mix.h5 --save-freq 1 --logdir test_line_td/linetd-1

