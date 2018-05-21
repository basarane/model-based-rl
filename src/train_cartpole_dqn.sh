# test_cartpole2/dqn-9'un tekrari network save parametreye alindi
rm test_cartpole2/dqn-15
python -B oo_dqn.py CartPole-v1 --model CartPoleModel --learning-rate 0.0003 --target-network-update 30 --egreedy-final 0.01 --egreedy-decay 0.999 --env-transforms Penalizer --replay-start-size 1000 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --max-step 30000 --save-interval 100 --logdir test_cartpole2/dqn-15


# use different trajectory sample epsilon = [0.0, 0.2, 0.4] trajectory_utils/TrajectoryLoader'da manuel kod degistirildi
#rm test_cartpole2/dqn-14 -rf
#python -B oo_dqn.py CartPole-v1 --model CartPoleModel --learning-rate 0.0003 --target-network-update 30 --egreedy-final 0.01 --egreedy-decay 0.1 --env-transforms Penalizer --replay-start-size 1000 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --max-step 100000 --load-trajectory test_tensorboard/traj-9_mix.h5 --logdir test_cartpole2/dqn-14

# use different trajectory sample epsilon = [0.6, 0.8, 1.0] trajectory_utils/TrajectoryLoader'da manuel kod degistirildi
#rm test_cartpole2/dqn-13 -rf
#python -B oo_dqn.py CartPole-v1 --model CartPoleModel --learning-rate 0.0003 --target-network-update 30 --egreedy-final 0.01 --egreedy-decay 0.1 --env-transforms Penalizer --replay-start-size 1000 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --max-step 100000 --load-trajectory test_tensorboard/traj-9_mix.h5 --logdir test_cartpole2/dqn-13

# use different trajectory sample epsilon = [0.4, 0.6, 0.8] trajectory_utils/TrajectoryLoader'da manuel kod degistirildi
#rm test_cartpole2/dqn-12 -rf
#python -B oo_dqn.py CartPole-v1 --model CartPoleModel --learning-rate 0.0003 --target-network-update 30 --egreedy-final 0.01 --egreedy-decay 0.1 --env-transforms Penalizer --replay-start-size 1000 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --max-step 100000 --load-trajectory test_tensorboard/traj-9_mix.h5 --logdir test_cartpole2/dqn-12

# use different trajectory sample epsilon = [0.2, 0.4, 0.6] trajectory_utils/TrajectoryLoader'da manuel kod degistirildi
#rm test_cartpole2/dqn-11 -rf
#python -B oo_dqn.py CartPole-v1 --model CartPoleModel --learning-rate 0.0003 --target-network-update 30 --egreedy-final 0.01 --egreedy-decay 0.1 --env-transforms Penalizer --replay-start-size 1000 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --max-step 100000 --load-trajectory test_tensorboard/traj-9_mix.h5 --logdir test_cartpole2/dqn-11

# use saved trajectories as replay buffer to test sample efficiency
#rm test_cartpole2/dqn-10 -rf
#python -B oo_dqn.py CartPole-v1 --model CartPoleModel --learning-rate 0.0003 --target-network-update 30 --egreedy-final 0.01 --egreedy-decay 0.1 --env-transforms Penalizer --replay-start-size 1000 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --max-step 100000 --load-trajectory test_tensorboard/traj-9_mix.h5 --logdir test_cartpole2/dqn-10

# StepEnv cikartildi, parametreler deney sonuclarina gore ayarlandi
#rm test_cartpole2/dqn-9 -rf
#python -B oo_dqn.py CartPole-v1 --model CartPoleModel --learning-rate 0.0003 --target-network-update 30 --egreedy-final 0.01 --egreedy-decay 0.999 --env-transforms Penalizer --replay-start-size 1000 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --max-step 100000 --logdir test_cartpole2/dqn-9

# StepEnv eklendikten sonra DQN tekrar train ediliyor
#rm test_cartpole2/dqn-8 -rf
#python -B oo_dqn.py CartPole-v1 --model CartPoleModel --learning-rate 0.0025 --target-network-update 200 --egreedy-final 0.01 --egreedy-decay 0.999 --env-transforms Penalizer StepEnv --replay-start-size 1000 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --max-step 100000 --logdir test_cartpole2/dqn-8


#rm test_tensorboard/traj5.h5
#python -B oo_dqn.py CartPole-v1 --model CartPoleModel --mode test --learning-rate 0.0025 --target-network-update 200 --egreedy-final 0.01 --egreedy-decay 0.999 --env-transforms Penalizer --replay-start-size 1000 --replay-buffer-size 0 --batch-size 64 --update-frequency 1 --max-step 50000 --test-epsilon 0.0 --render-step 1 --load-weightfile test_cartpole2/dqn-7/weights_50000.h5 --save-trajectory test_tensorboard/traj5.h5

#rm test_tensorboard/traj4.h5
#python -B oo_dqn.py CartPole-v1 --model CartPoleModel --mode test --learning-rate 0.0025 --target-network-update 200 --egreedy-final 0.01 --egreedy-decay 0.999 --env-transforms Penalizer --replay-start-size 1000 --replay-buffer-size 0 --batch-size 64 --update-frequency 1 --max-step 50000 --test-epsilon 0.0 --render-step 1 --load-weightfile test_cartpole2/dqn-7/weights_50000.h5 --save-trajectory test_tensorboard/traj4.h5

#rm test_tensorboard/traj3.h5
#python -B oo_dqn.py CartPole-v1 --model CartPoleModel --mode test --learning-rate 0.0025 --target-network-update 200 --egreedy-final 0.01 --egreedy-decay 0.999 --env-transforms Penalizer --replay-start-size 1000 --replay-buffer-size 0 --batch-size 64 --update-frequency 1 --max-step 50000 --test-epsilon 0.1 --render-step 1 --load-weightfile test_cartpole2/dqn-7/weights_50000.h5 --save-trajectory test_tensorboard/traj3.h5

#rm test_tensorboard/traj2.h5
#python -B oo_dqn.py CartPole-v1 --model CartPoleModel --mode test --learning-rate 0.0025 --target-network-update 200 --egreedy-final 0.01 --egreedy-decay 0.999 --env-transforms Penalizer --replay-start-size 1000 --replay-buffer-size 0 --batch-size 64 --update-frequency 1 --max-step 50000 --test-epsilon 0.5 --render-step 1 --load-weightfile test_cartpole2/dqn-7/weights_50000.h5 --save-trajectory test_tensorboard/traj2.h5

# Son run'in testi icin
#rm test_tensorboard/traj.h5
#python -B oo_dqn.py CartPole-v1 --model CartPoleModel --mode test --learning-rate 0.0025 --target-network-update 200 --egreedy-final 0.01 --egreedy-decay 0.999 --env-transforms Penalizer --replay-start-size 1000 --replay-buffer-size 0 --batch-size 64 --update-frequency 1 --max-step 50000 --test-epsilon 0.3 --render-step 1 --load-weightfile test_cartpole2/dqn-7/weights_50000.h5 --save-trajectory test_tensorboard/traj.h5

#python -B oo_dqn.py CartPole-v1 --model CartPoleModel --learning-rate 0.0025 --target-network-update 200 --egreedy-final 0.01 --egreedy-decay 0.999 --env-transforms Penalizer --replay-start-size 1000 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --max-step 50000 --logdir test_cartpole2/dqn-7

#python -B oo_dqn.py CartPole-v1 --model CartPoleModel --learning-rate 0.01 --target-network-update 200 --egreedy-final 0.01 --egreedy-decay 0.999 --env-transforms Penalizer --replay-start-size 1000 --replay-buffer-size 20000 --batch-size 64 --update-frequency 1 --max-step 50000 --logdir test_cartpole2/dqn-6

#python -B oo_dqn.py CartPole-v1 --model CartPoleModel --learning-rate 0.001 --target-network-update 200 --egreedy-final 0.01 --egreedy-decay 0.999 --env-transforms Penalizer --replay-start-size 1000 --replay-buffer-size 20000 --batch-size 64 --update-frequency 1 --max-step 50000 --logdir test_cartpole2/dqn-5

#python -B oo_dqn.py CartPole-v1 --model CartPoleModel --learning-rate 0.001 --target-network-update 20 --egreedy-final 0.01 --egreedy-decay 0.999 --env-transforms Penalizer --replay-start-size 1000 --replay-buffer-size 20000 --batch-size 64 --update-frequency 1 --max-step 50000 --logdir test_cartpole2/dqn-4

#python -B oo_a3c.py CartPole-v1 --model CartPoleModel --nstep 5 --learning-rate 0.01 --target-network-update 20 --egreedy-final 0.01 --egreedy-decay 0.9999 0.999 0.998 --egreedy-props 0.4 0.3 0.3 --env-transforms Penalizer --replay-start-size 64 --replay-buffer-size 2000 --batch-size 64 --update-frequency 1 --thread-count 16 --max-step 200000 --logdir test_cartpole2/a3c-10

# the following lines are tests

#python -B oo_a3c.py CartPole-v1 --model CartPoleModel --nstep 5 --learning-rate 0.001 --target-network-update 20 --egreedy-final 0.001 --egreedy-props 1 --egreedy-final-step 6000 --egreedy-decay 1 --env-transforms Penalizer --replay-start-size 64 --replay-buffer-size 2000 --batch-size 64 --update-frequency 1 --thread-count 16 --logdir test_cartpole/a3c-t16-n5-ed1-efs6000-8

#python -B oo_a3c.py CartPole-v1 --model CartPoleModel --nstep 0 --learning-rate 0.001 --target-network-update 20 --egreedy-final 0.01 --egreedy-decay 0.995 --env-transforms Penalizer --replay-start-size 64 --replay-buffer-size 2000 --batch-size 64 --update-frequency 1 --thread-count 16 --logdir test_cartpole/a3c-t16-n0-ed0.998-7
