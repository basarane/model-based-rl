python -B oo_dqn.py CartPole-v1 --model CartPoleModel --learning-rate 0.001 --target-network-update 20 --egreedy-final 0.01 --egreedy-decay 0.999 --env-transforms Penalizer --replay-start-size 1000 --replay-buffer-size 20000 --batch-size 64 --update-frequency 1 --max-step 50000 --logdir test_cartpole2/dqn-4

#python -B oo_a3c.py CartPole-v1 --model CartPoleModel --nstep 5 --learning-rate 0.01 --target-network-update 20 --egreedy-final 0.01 --egreedy-decay 0.9999 0.999 0.998 --egreedy-props 0.4 0.3 0.3 --env-transforms Penalizer --replay-start-size 64 --replay-buffer-size 2000 --batch-size 64 --update-frequency 1 --thread-count 16 --max-step 200000 --logdir test_cartpole2/a3c-10

# the following lines are tests

#python -B oo_a3c.py CartPole-v1 --model CartPoleModel --nstep 5 --learning-rate 0.001 --target-network-update 20 --egreedy-final 0.001 --egreedy-props 1 --egreedy-final-step 6000 --egreedy-decay 1 --env-transforms Penalizer --replay-start-size 64 --replay-buffer-size 2000 --batch-size 64 --update-frequency 1 --thread-count 16 --logdir test_cartpole/a3c-t16-n5-ed1-efs6000-8

#python -B oo_a3c.py CartPole-v1 --model CartPoleModel --nstep 0 --learning-rate 0.001 --target-network-update 20 --egreedy-final 0.01 --egreedy-decay 0.995 --env-transforms Penalizer --replay-start-size 64 --replay-buffer-size 2000 --batch-size 64 --update-frequency 1 --thread-count 16 --logdir test_cartpole/a3c-t16-n0-ed0.998-7
