# sadece train suresi uzatildi: 300k
rm test_cartpole_model3/reward-27 -rf
python -B oo_env.py CartPole-v1 --mode train --env-model EnvModelCartpole --env-transforms Penalizer --model CartPoleModel --learning-rate 0.001 --batch-size 64 --max-step 300000 --save-interval 5000 --load-trajectory test_tensorboard/traj-9_mix.h5 --logdir test_cartpole_model3/reward-27

# sadece -100 durumunda done = 1 yapildi, diger durumlarda done = 0 kabul ediliyor # dones = [1 if a['reward']<0 else 0 for a in samples]
#rm test_cartpole_model3/reward-26 -rf
#python -B oo_env.py CartPole-v1 --mode train --env-model EnvModelCartpole --env-transforms Penalizer --model CartPoleModel --learning-rate 0.001 --batch-size 64 --max-step 100000 --save-interval 5000 --load-trajectory test_tensorboard/traj-9_mix.h5 --logdir test_cartpole_model3/reward-26

# StepEnv tekrar cikartildi, DQN optimum hyper parametrelerle calistirildi
#rm test_cartpole_model3/reward-25 -rf
#python -B oo_env.py CartPole-v1 --mode train --env-model EnvModelCartpole --env-transforms Penalizer --model CartPoleModel --learning-rate 0.001 --batch-size 64 --max-step 100000 --save-interval 5000 --load-trajectory test_tensorboard/traj-9_mix.h5 --logdir test_cartpole_model3/reward-25

# bir onceki ile ayni, sadece TrajectoryLoader'da guncelleme yaptim, herseyi memory'e nump arrayi olarak yukluyor init'de, onceden her sample'da yapiyordu. x7 hizlandi :))
#rm test_cartpole_model3/reward-24 -rf
#python -B oo_env.py CartPole-v1 --mode train --env-transforms Penalizer StepEnv --model CartPoleModel --learning-rate 0.001 --batch-size 64 --max-step 100000 --save-interval 5000 --load-trajectory test_tensorboard/traj-8_mix.h5 --logdir test_cartpole_model3/reward-24

# son bir kac runi unut. dqn'i StepEnv ekledikten sonra tekrar calistirmam gerekti
#rm test_cartpole_model3/reward-23 -rf
#python -B oo_env.py CartPole-v1 --mode train --env-transforms Penalizer StepEnv --model CartPoleModel --learning-rate 0.001 --batch-size 64 --max-step 100000 --save-interval 5000 --load-trajectory test_tensorboard/traj-8_mix.h5 --logdir test_cartpole_model3/reward-23

# onceki run'in aynisi sanirim :) ama TrajectoryReplay ve TrajRunner class'larini yeni yazmistim halbuki 
#rm test_cartpole_model2/reward-22 -rf
#python -B oo_env.py CartPole-v1 --mode train --model CartPoleModel --learning-rate 0.001 --batch-size 64 --max-step 100000 --save-interval 5000 --load-trajectory test_tensorboard/traj_mix.h5 --logdir test_cartpole_model2/reward-22

# step count state'e eklendi (StepEnv), data traj.h5 okunacak sekilde degistirildi
#rm test_cartpole_model2/reward-21 -rf
#python -B oo_env.py CartPole-v1 --mode train --model CartPoleModel --learning-rate 0.001 --env-transforms Penalizer --replay-start-size 64 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --max-step 100000 --load-weightfile test_cartpole2/dqn-7/weights_50000.h5 --test-epsilon 0.3 --reward-scale 0.01 --save-interval 5000 --logdir test_cartpole_model2/reward-21

#rm test_cartpole_model2/reward-20 -rf
#python -B oo_env.py CartPole-v1 --mode train --model CartPoleModel --learning-rate 0.001 --env-transforms Penalizer --replay-start-size 64 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --max-step 100000 --load-weightfile test_cartpole2/dqn-7/weights_50000.h5 --test-epsilon 0.6 --reward-scale 0.01 --save-interval 5000 --logdir test_cartpole_model2/reward-20

# my_optimizer = Adam(lr=self.ops.LEARNING_RATE)
#python -B oo_env.py CartPole-v1 --mode train --model CartPoleModel --learning-rate 0.001 --env-transforms Penalizer --replay-start-size 64 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --max-step 100000 --load-weightfile test_cartpole2/dqn-7/weights_50000.h5 --test-epsilon 0.6 --reward-scale 0.01 --save-interval 5000 --logdir test_cartpole_model2/reward-19

# my_optimizer = RMSprop(lr=self.ops.LEARNING_RATE, rho=0.90, decay=0.0) #epsilon=None, 
#python -B oo_env.py CartPole-v1 --mode train --model CartPoleModel --learning-rate 0.001 --env-transforms Penalizer --replay-start-size 64 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --max-step 100000 --load-weightfile test_cartpole2/dqn-7/weights_50000.h5 --test-epsilon 0.6 --reward-scale 0.01 --save-interval 5000 --logdir test_cartpole_model2/reward-18


# alttaki neydi hatirlayamadim
#python -B oo_env.py CartPole-v1 --mode train --model CartPoleModel --learning-rate 0.0001 --env-transforms Penalizer --replay-start-size 64 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --max-step 50000 --load-weightfile test_cartpole2/dqn-7/weights_50000.h5 --test-epsilon 0.6 --reward-scale 0.01 


# SON EGITILEN MODEL 
#python -B oo_env.py CartPole-v1 --mode test --model CartPoleModel --learning-rate 0.0001 --env-transforms Penalizer --replay-start-size 64 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --max-step 500000 --load-weightfile test_cartpole2/dqn-7/weights_50000.h5 --test-epsilon 0.6 --reward-scale 0.01 --save-interval 5000 --logdir test_cartpole_model/reward-17

# reward-15 tensorboard'a bos geldi, ayni parametrelerle
#python -B oo_env.py CartPole-v1 --mode test --model CartPoleModel --learning-rate 0.00025 --env-transforms Penalizer --replay-start-size 64 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --max-step 50000 --load-weightfile test_cartpole2/dqn-7/weights_50000.h5 --test-epsilon 0.6 --reward-scale 0.01 --save-interval 5000 --logdir test_cartpole_model/reward-16

# save added
#python -B oo_env.py CartPole-v1 --mode test --model CartPoleModel --learning-rate 0.002 --env-transforms Penalizer --replay-start-size 64 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --max-step 50000 --load-weightfile test_cartpole2/dqn-7/weights_50000.h5 --test-epsilon 0.6 --reward-scale 0.01 --save-interval 5000 --logdir test_cartpole_model/reward-14

#only reward is clipped (-1, +1), reward_scale is not applied reward-12 is the same (may be without save??)
#python -B oo_env.py CartPole-v1 --mode test --model CartPoleModel --learning-rate 0.002 --env-transforms Penalizer --replay-start-size 64 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --max-step 50000 --load-weightfile test_cartpole2/dqn-7/weights_50000.h5 --test-epsilon 0.6 --reward-scale 0.01 --logdir test_cartpole_model/reward-13

#previous run has error, corrected: ac = len(est_next_states)-1
#python -B oo_env.py CartPole-v1 --mode test --model CartPoleModel --learning-rate 0.002 --env-transforms Penalizer --replay-start-size 64 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --max-step 50000 --load-weightfile test_cartpole2/dqn-7/weights_50000.h5 --test-epsilon 0.6 --reward-scale 0.01 --logdir test_cartpole_model/reward-11

#network 256x256xACTION, seperate reward output in single vector
#python -B oo_env.py CartPole-v1 --mode test --model CartPoleModel --learning-rate 0.002 --env-transforms Penalizer --replay-start-size 64 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --max-step 50000 --load-weightfile test_cartpole2/dqn-7/weights_50000.h5 --test-epsilon 0.6 --reward-scale 0.01 --logdir test_cartpole_model/reward-10

#network 256x256xACTION, seperate reward outputs, random uniform (-0.02, 0.02), added 256 hidden units for reward 
#python -B oo_env.py CartPole-v1 --mode test --model CartPoleModel --learning-rate 0.002 --env-transforms Penalizer --replay-start-size 64 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --max-step 50000 --load-weightfile test_cartpole2/dqn-7/weights_50000.h5 --test-epsilon 0.6 --reward-scale 0.01 --logdir test_cartpole_model/reward-9

#network 256x256xACTION, seperate reward outputs, random uniform (-0.02, 0.02), reward set to zero to test loss
#python -B oo_env.py CartPole-v1 --mode test --model CartPoleModel --learning-rate 0.002 --env-transforms Penalizer --replay-start-size 64 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --max-step 50000 --load-weightfile test_cartpole2/dqn-7/weights_50000.h5 --test-epsilon 0.6 --reward-scale 0.01 --logdir test_cartpole_model/reward-8

#network 256x256xACTION, seperate reward outputs, random uniform (-0.02, 0.02)
#python -B oo_env.py CartPole-v1 --mode test --model CartPoleModel --learning-rate 0.002 --env-transforms Penalizer --replay-start-size 64 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --max-step 50000 --load-weightfile test_cartpole2/dqn-7/weights_50000.h5 --test-epsilon 0.6 --reward-scale 0.01 --logdir test_cartpole_model/reward-7

#network 256x256xACTION, seperate reward outputs
#python -B oo_env.py CartPole-v1 --mode test --model CartPoleModel --learning-rate 0.002 --env-transforms Penalizer --replay-start-size 64 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --max-step 50000 --load-weightfile test_cartpole2/dqn-7/weights_50000.h5 --test-epsilon 0.6 --reward-scale 0.01 --logdir test_cartpole_model/reward-6

#network 256x256x256xACTION
#python -B oo_env.py CartPole-v1 --mode test --model CartPoleModel --learning-rate 0.005 --env-transforms Penalizer --replay-start-size 64 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --max-step 50000 --load-weightfile test_cartpole2/dqn-7/weights_50000.h5 --test-epsilon 0.6 --reward-scale 0.01 --logdir test_cartpole_model/reward-5

#network 256x256x256xACTION
#python -B oo_env.py CartPole-v1 --mode test --model CartPoleModel --learning-rate 0.001 --env-transforms Penalizer --replay-start-size 64 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --max-step 50000 --load-weightfile test_cartpole2/dqn-7/weights_50000.h5 --test-epsilon 0.6 --reward-scale 0.01 --logdir test_cartpole_model/reward-4

#network 256x256xACTION
#python -B oo_env.py CartPole-v1 --mode test --model CartPoleModel --learning-rate 0.001 --env-transforms Penalizer --replay-start-size 64 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --max-step 50000 --load-weightfile test_cartpole2/dqn-7/weights_50000.h5 --test-epsilon 0.6 --reward-scale 0.01 --logdir test_cartpole_model/reward-3

# network: 30x30xACTION
#python -B oo_env.py CartPole-v1 --mode test --model CartPoleModel --learning-rate 0.001 --env-transforms Penalizer --replay-start-size 64 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --max-step 50000 --load-weightfile test_cartpole2/dqn-7/weights_50000.h5 --test-epsilon 0.6 --reward-scale 0.01 --logdir test_cartpole_model/reward-2

#python -B oo_env.py CartPole-v1 --mode test --model CartPoleModel --learning-rate 0.001 --env-transforms Penalizer --replay-start-size 64 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --max-step 50000 --load-weightfile test_cartpole2/dqn-7/weights_50000.h5 --test-epsilon 0.6 --logdir test_cartpole_model/reward-1

#python -B oo_env.py CartPole-v1 --mode test --model CartPoleModel --learning-rate 0.001 --env-transforms Penalizer --replay-start-size 64 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --max-step 50000 --load-weightfile test_cartpole2/dqn-7/weights_50000.h5 --test-epsilon 0.6 --logdir test_cartpole_model/e0.6-1

#python -B oo_env.py CartPole-v1 --mode test --model CartPoleModel --learning-rate 0.0025 --env-transforms Penalizer --replay-start-size 64 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --max-step 50000 --load-weightfile test_cartpole2/dqn-7/weights_50000.h5 --test-epsilon 0.4 --logdir test_cartpole_model/e0.4-1

#python -B oo_env.py CartPole-v1 --mode test --model CartPoleModel --learning-rate 0.0025 --env-transforms Penalizer --replay-start-size 64 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --max-step 50000 --load-weightfile test_cartpole2/dqn-7/weights_50000.h5 --test-epsilon 0.2 --logdir test_cartpole_model/e0.2-1

#--enable-render False --render-step 4 