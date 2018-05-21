#rm test_tensorboard/traj-9b_mix.h5
#python -B oo_dqn.py CartPole-v1 --model CartPoleModel --mode test --learning-rate 0.0025 --target-network-update 200 --egreedy-final 0.01 --egreedy-decay 0.999 --env-transforms Penalizer --replay-start-size 1000 --replay-buffer-size 0 --batch-size 64 --update-frequency 1 --max-step 50000 --test-epsilon 0.0 --render-step 1 --load-weightfile test_cartpole2/dqn-9/weights_100000.h5 --save-trajectory test_tensorboard/traj-9b_mix.h5
#python -B oo_dqn.py CartPole-v1 --model CartPoleModel --mode test --learning-rate 0.0025 --target-network-update 200 --egreedy-final 0.01 --egreedy-decay 0.999 --env-transforms Penalizer --replay-start-size 1000 --replay-buffer-size 0 --batch-size 64 --update-frequency 1 --max-step 50000 --test-epsilon 0.4 --render-step 1 --load-weightfile test_cartpole2/dqn-9/weights_100000.h5 --save-trajectory test_tensorboard/traj-9b_mix.h5

rm test_tensorboard/traj-9_mix.h5
python -B oo_dqn.py CartPole-v1 --model CartPoleModel --mode test --learning-rate 0.0025 --target-network-update 200 --egreedy-final 0.01 --egreedy-decay 0.999 --env-transforms Penalizer --replay-start-size 1000 --replay-buffer-size 0 --batch-size 64 --update-frequency 1 --max-step 50000 --test-epsilon 0.0 --render-step 1 --load-weightfile test_cartpole2/dqn-9/weights_100000.h5 --save-trajectory test_tensorboard/traj-9_mix.h5
python -B oo_dqn.py CartPole-v1 --model CartPoleModel --mode test --learning-rate 0.0025 --target-network-update 200 --egreedy-final 0.01 --egreedy-decay 0.999 --env-transforms Penalizer --replay-start-size 1000 --replay-buffer-size 0 --batch-size 64 --update-frequency 1 --max-step 50000 --test-epsilon 0.2 --render-step 1 --load-weightfile test_cartpole2/dqn-9/weights_100000.h5 --save-trajectory test_tensorboard/traj-9_mix.h5
python -B oo_dqn.py CartPole-v1 --model CartPoleModel --mode test --learning-rate 0.0025 --target-network-update 200 --egreedy-final 0.01 --egreedy-decay 0.999 --env-transforms Penalizer --replay-start-size 1000 --replay-buffer-size 0 --batch-size 64 --update-frequency 1 --max-step 50000 --test-epsilon 0.4 --render-step 1 --load-weightfile test_cartpole2/dqn-9/weights_100000.h5 --save-trajectory test_tensorboard/traj-9_mix.h5
python -B oo_dqn.py CartPole-v1 --model CartPoleModel --mode test --learning-rate 0.0025 --target-network-update 200 --egreedy-final 0.01 --egreedy-decay 0.999 --env-transforms Penalizer --replay-start-size 1000 --replay-buffer-size 0 --batch-size 64 --update-frequency 1 --max-step 50000 --test-epsilon 0.6 --render-step 1 --load-weightfile test_cartpole2/dqn-9/weights_100000.h5 --save-trajectory test_tensorboard/traj-9_mix.h5
python -B oo_dqn.py CartPole-v1 --model CartPoleModel --mode test --learning-rate 0.0025 --target-network-update 200 --egreedy-final 0.01 --egreedy-decay 0.999 --env-transforms Penalizer --replay-start-size 1000 --replay-buffer-size 0 --batch-size 64 --update-frequency 1 --max-step 50000 --test-epsilon 0.8 --render-step 1 --load-weightfile test_cartpole2/dqn-9/weights_100000.h5 --save-trajectory test_tensorboard/traj-9_mix.h5
python -B oo_dqn.py CartPole-v1 --model CartPoleModel --mode test --learning-rate 0.0025 --target-network-update 200 --egreedy-final 0.01 --egreedy-decay 0.999 --env-transforms Penalizer --replay-start-size 1000 --replay-buffer-size 0 --batch-size 64 --update-frequency 1 --max-step 50000 --test-epsilon 1 --render-step 1 --load-weightfile test_cartpole2/dqn-9/weights_100000.h5 --save-trajectory test_tensorboard/traj-9_mix.h5

#rm test_tensorboard/traj-8_mix.h5
#python -B oo_dqn.py CartPole-v1 --model CartPoleModel --mode test --learning-rate 0.0025 --target-network-update 200 --egreedy-final 0.01 --egreedy-decay 0.999 --env-transforms Penalizer StepEnv --replay-start-size 1000 --replay-buffer-size 0 --batch-size 64 --update-frequency 1 --max-step 50000 --test-epsilon 0.0 --render-step 1 --load-weightfile test_cartpole2/dqn-8/weights_100000.h5 --save-trajectory test_tensorboard/traj-8_mix.h5
#python -B oo_dqn.py CartPole-v1 --model CartPoleModel --mode test --learning-rate 0.0025 --target-network-update 200 --egreedy-final 0.01 --egreedy-decay 0.999 --env-transforms Penalizer StepEnv --replay-start-size 1000 --replay-buffer-size 0 --batch-size 64 --update-frequency 1 --max-step 50000 --test-epsilon 0.2 --render-step 1 --load-weightfile test_cartpole2/dqn-8/weights_100000.h5 --save-trajectory test_tensorboard/traj-8_mix.h5
#python -B oo_dqn.py CartPole-v1 --model CartPoleModel --mode test --learning-rate 0.0025 --target-network-update 200 --egreedy-final 0.01 --egreedy-decay 0.999 --env-transforms Penalizer StepEnv --replay-start-size 1000 --replay-buffer-size 0 --batch-size 64 --update-frequency 1 --max-step 50000 --test-epsilon 0.4 --render-step 1 --load-weightfile test_cartpole2/dqn-8/weights_100000.h5 --save-trajectory test_tensorboard/traj-8_mix.h5
#python -B oo_dqn.py CartPole-v1 --model CartPoleModel --mode test --learning-rate 0.0025 --target-network-update 200 --egreedy-final 0.01 --egreedy-decay 0.999 --env-transforms Penalizer StepEnv --replay-start-size 1000 --replay-buffer-size 0 --batch-size 64 --update-frequency 1 --max-step 50000 --test-epsilon 0.6 --render-step 1 --load-weightfile test_cartpole2/dqn-8/weights_100000.h5 --save-trajectory test_tensorboard/traj-8_mix.h5
#python -B oo_dqn.py CartPole-v1 --model CartPoleModel --mode test --learning-rate 0.0025 --target-network-update 200 --egreedy-final 0.01 --egreedy-decay 0.999 --env-transforms Penalizer StepEnv --replay-start-size 1000 --replay-buffer-size 0 --batch-size 64 --update-frequency 1 --max-step 50000 --test-epsilon 0.8 --render-step 1 --load-weightfile test_cartpole2/dqn-8/weights_100000.h5 --save-trajectory test_tensorboard/traj-8_mix.h5
#python -B oo_dqn.py CartPole-v1 --model CartPoleModel --mode test --learning-rate 0.0025 --target-network-update 200 --egreedy-final 0.01 --egreedy-decay 0.999 --env-transforms Penalizer StepEnv --replay-start-size 1000 --replay-buffer-size 0 --batch-size 64 --update-frequency 1 --max-step 50000 --test-epsilon 1 --render-step 1 --load-weightfile test_cartpole2/dqn-8/weights_100000.h5 --save-trajectory test_tensorboard/traj-8_mix.h5
