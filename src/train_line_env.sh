# sadece train suresi uzatildi: 300k
rm test_line_env/lineenv-1 -rf
python -B oo_env.py Line --mode train --env-model EnvModelLine --model LineModel --learning-rate 0.001 --batch-size 64 --max-step 5000 --save-interval 100 --load-trajectory test_line_dqn/traj-2_mix.h5 --logdir test_line_env/lineenv-1
