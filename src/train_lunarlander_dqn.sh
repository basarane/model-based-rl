# 512 hidden unit
python -B oo_dqn.py LunarLander-v2 --model LunarLanderModel --learning-rate 0.00025 --target-network-update 5000 --egreedy-final 0.01 --egreedy-final-step 250000 --egreedy-decay 1 --replay-start-size 5000 --replay-buffer-size 50000 --batch-size 64 --update-frequency 1 --max-step 1000000 --logdir test_lunarlander/dqn-9
#python -B oo_dqn.py LunarLander-v2 --mode test --test-epsilon 0.00 --enable-render True --render-step 10 --load-weightfile test_lunarlander/dqn-9/weights_750000.h5 --model LunarLanderModel --learning-rate 0.00025 --target-network-update 5000 --egreedy-final 0.01 --egreedy-final-step 250000 --egreedy-decay 1 --replay-start-size 5000 --replay-buffer-size 50000 --batch-size 64 --update-frequency 1 --max-step 1000000 



# 1024 hidden unit
#python -B oo_dqn.py LunarLander-v2 --model LunarLanderModel --learning-rate 0.00025 --target-network-update 500 --egreedy-final 0.05 --egreedy-final-step 100000 --egreedy-decay 1 --replay-start-size 1000 --replay-buffer-size 20000 --batch-size 64 --update-frequency 1 --max-step 200000 --logdir test_lunarlander/dqn-8

# 256 hidden unit
#python -B oo_dqn.py LunarLander-v2 --model LunarLanderModel --learning-rate 0.00025 --target-network-update 500 --egreedy-final 0.05 --egreedy-final-step 100000 --egreedy-decay 1 --replay-start-size 1000 --replay-buffer-size 20000 --batch-size 64 --update-frequency 1 --max-step 200000 --logdir test_lunarlander/dqn-7

#python -B oo_dqn.py LunarLander-v2 --model LunarLanderModel --learning-rate 0.0005 --target-network-update 20 --egreedy-final 0.05 --egreedy-final-step 100000 --egreedy-decay 1 --replay-start-size 1000 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --max-step 200000 --logdir test_lunarlander/dqn-6

#python -B oo_dqn.py LunarLander-v2 --model CartPoleModel --learning-rate 0.0005 --target-network-update 20 --egreedy-final 0.05 --egreedy-final-step 100000 --egreedy-decay 1 --replay-start-size 1000 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --max-step 200000 --logdir test_lunarlander/dqn-5

#python -B oo_a3c.py CartPole-v1 --model CartPoleModel --nstep 5 --learning-rate 0.01 --target-network-update 20 --egreedy-final 0.01 --egreedy-decay 0.9999 0.999 0.998 --egreedy-props 0.4 0.3 0.3 --env-transforms Penalizer --replay-start-size 64 --replay-buffer-size 2000 --batch-size 64 --update-frequency 1 --thread-count 16 --max-step 200000 --logdir test_cartpole2/a3c-10

# the following lines are tests

#python -B oo_a3c.py CartPole-v1 --model CartPoleModel --nstep 5 --learning-rate 0.001 --target-network-update 20 --egreedy-final 0.001 --egreedy-props 1 --egreedy-final-step 6000 --egreedy-decay 1 --env-transforms Penalizer --replay-start-size 64 --replay-buffer-size 2000 --batch-size 64 --update-frequency 1 --thread-count 16 --logdir test_cartpole/a3c-t16-n5-ed1-efs6000-8

#python -B oo_a3c.py CartPole-v1 --model CartPoleModel --nstep 0 --learning-rate 0.001 --target-network-update 20 --egreedy-final 0.01 --egreedy-decay 0.995 --env-transforms Penalizer --replay-start-size 64 --replay-buffer-size 2000 --batch-size 64 --update-frequency 1 --thread-count 16 --logdir test_cartpole/a3c-t16-n0-ed0.998-7
