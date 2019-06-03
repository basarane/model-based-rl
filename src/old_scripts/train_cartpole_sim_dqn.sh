# reward'lari da envmodel'den al
#rm test_cartpole_dqn_sim/dqnsim-4 -rf
#python -B oo_dqn.py CartPole-v1 --model CartPoleModel --learning-rate 0.0003 --target-network-update 30 --egreedy-final 0.01 --egreedy-decay 0.999 --env-transforms Penalizer --replay-start-size 1000 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --max-step 100000 --env-model EnvModelCartpole --env-weightfile test_cartpole_model3/reward-25/weights_100000.h5 --env-reward True --logdir test_cartpole_dqn_sim/dqnsim-4

# iki onceki envmodel ile test
#rm test_cartpole_dqn_sim/dqnsim-3 -rf
#python -B oo_dqn.py CartPole-v1 --model CartPoleModel --learning-rate 0.0003 --target-network-update 30 --egreedy-final 0.01 --egreedy-decay 0.999 --env-transforms Penalizer --replay-start-size 1000 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --max-step 100000 --env-model EnvModelCartpole --env-weightfile test_cartpole_model3/reward-25/weights_100000.h5 --logdir test_cartpole_dqn_sim/dqnsim-3

# en son envmodel ile test
#rm test_cartpole_dqn_sim/dqnsim-2 -rf
#python -B oo_dqn.py CartPole-v1 --model CartPoleModel --learning-rate 0.0003 --target-network-update 30 --egreedy-final 0.01 --egreedy-decay 0.999 --env-transforms Penalizer --replay-start-size 1000 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --max-step 100000 --env-model EnvModelCartpole --env-weightfile test_cartpole_model3/reward-27/weights_300000.h5 --logdir test_cartpole_dqn_sim/dqnsim-2

# Ilk sim test
rm test_cartpole_dqn_sim/dqnsim-1 -rf
python -B oo_dqn.py CartPole-v1 --model CartPoleModel --learning-rate 0.0003 --target-network-update 30 --egreedy-final 0.01 --egreedy-decay 0.999 --env-transforms Penalizer --replay-start-size 1000 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --max-step 100000 --env-model EnvModelCartpole --env-weightfile test_cartpole_model3/reward-26/weights_100000.h5 --logdir test_cartpole_dqn_sim/dqnsim-1
