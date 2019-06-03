rm test_cartpole_td/test39 -rf
python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model3/reward-25/weights_100000.h5 --batch-size 64 --max-step 1000 --learning-rate 0.001 --target-network-update 500000 --env-model EnvModelCartpole --vmodel CartPoleVNetwork  --load-trajectory test_tensorboard/traj-9_mix.h5 --save-freq 1 --logdir test_cartpole_td/test39

# td-test30, v-test14 tekrari x 1000 step
# use different trajectory sample epsilon = [0.2, 0.4, 0.6] trajectory_utils/TrajectoryLoader'da manuel kod degistirildi
#rm test_cartpole_td/test38 -rf
#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model3/reward-25/weights_100000.h5 --batch-size 64 --max-step 1000 --learning-rate 0.001 --target-network-update 500000 --env-model EnvModelCartpole --vmodel CartPoleVNetwork  --load-trajectory test_tensorboard/traj-9_mix.h5 --save-freq 10 --logdir test_cartpole_td/test38

# td-test30, v-test14 tekrari x 1000 step
#rm test_cartpole_td/test37 -rf
#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model3/reward-25/weights_100000.h5 --batch-size 64 --max-step 1000 --learning-rate 0.001 --target-network-update 500000 --env-model EnvModelCartpole --vmodel CartPoleVNetwork  --load-trajectory test_tensorboard/traj-9_mix.h5 --save-freq 10 --logdir test_cartpole_td/test37

# td-test30, v-test14 tekrari x 100 step
#rm test_cartpole_td/test36 -rf
#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model3/reward-25/weights_100000.h5 --batch-size 64 --max-step 100 --learning-rate 0.001 --target-network-update 500000 --env-model EnvModelCartpole --vmodel CartPoleVNetwork  --load-trajectory test_tensorboard/traj-9_mix.h5 --save-freq 1 --logdir test_cartpole_td/test36

# test32'nin tekrari
#rm test_cartpole_td/test35 -rf
#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model3/reward-25/weights_100000.h5 --batch-size 64 --max-step 100000 --learning-rate 0.00025 --target-network-update 500000 --env-model EnvModelCartpole --vmodel CartPoleVNetwork  --load-trajectory test_tensorboard/traj-9_mix.h5 --save-freq 1000 --logdir test_cartpole_td/test35

# train_cartpole_sim_dqn.sh ile test ettigimde en iyi sonucu veren env ile calistirdim - EN IYI SONUCU VERMIYORMUS, YANLIS BILGI
#rm test_cartpole_td/test34 -rf
#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model3/reward-27/weights_300000.h5 --batch-size 64 --max-step 100000 --learning-rate 0.00025 --target-network-update 500000 --env-model EnvModelCartpole --vmodel CartPoleVNetwork  --load-trajectory test_tensorboard/traj-9_mix.h5 --save-freq 1000 --logdir test_cartpole_td/test34

# envmodel'de reward<0 => done=1 yapildi
#rm test_cartpole_td/test33 -rf
#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model3/reward-26/weights_100000.h5 --batch-size 64 --max-step 100000 --learning-rate 0.00025 --target-network-update 500000 --env-model EnvModelCartpole --vmodel CartPoleVNetwork  --load-trajectory test_tensorboard/traj-9_mix.h5 --save-freq 1000 --logdir test_cartpole_td/test33

# RMSprop'dan ADAM'a gecildi
#rm test_cartpole_td/test32 -rf
#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model3/reward-25/weights_100000.h5 --batch-size 64 --max-step 250000 --learning-rate 0.00025 --target-network-update 500000 --env-model EnvModelCartpole --vmodel CartPoleVNetwork  --load-trajectory test_tensorboard/traj-9_mix.h5 --save-freq 1000 --logdir test_cartpole_td/test32

# learning rate'i iyice dusurdum
#rm test_cartpole_td/test31 -rf
#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model3/reward-25/weights_100000.h5 --batch-size 64 --max-step 250000 --learning-rate 0.0001 --target-network-update 500000 --env-model EnvModelCartpole --vmodel CartPoleVNetwork  --load-trajectory test_tensorboard/traj-9_mix.h5 --save-freq 1000 --logdir test_cartpole_td/test31

# bir oncekinin hyperparametre testleri - learning rate'i dusurdum
#rm test_cartpole_td/test30 -rf
#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model3/reward-25/weights_100000.h5 --batch-size 64 --max-step 250000 --learning-rate 0.001 --target-network-update 500000 --env-model EnvModelCartpole --vmodel CartPoleVNetwork  --load-trajectory test_tensorboard/traj-9_mix.h5 --save-freq 1000 --logdir test_cartpole_td/test30

# bir onceki ile ayni, save_freq eklendi, sonrasinda farkli step'lerdeki V networkleri test edilecek
#rm test_cartpole_td/test29 -rf
#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model3/reward-25/weights_100000.h5 --batch-size 64 --max-step 250000 --learning-rate 0.005 --target-network-update 500000 --env-model EnvModelCartpole --vmodel CartPoleVNetwork  --load-trajectory test_tensorboard/traj-9_mix.h5 --save-freq 1000 --logdir test_cartpole_td/test29

# StepEnv tekrar cikartildi, td de V done ile carpiliyordu 1-done ile duzeltildi (ama bu run'da varmi tam emin olamadim)
#rm test_cartpole_td/test28 -rf
#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model3/reward-25/weights_100000.h5 --batch-size 64 --max-step 100000 --learning-rate 0.005 --target-network-update 500000 --env-model EnvModelCartpole --vmodel CartPoleVNetwork  --load-trajectory test_tensorboard/traj-9_mix.h5 --logdir test_cartpole_td/test28

# multiple epsilon dqn agents [0, 0.2, 0.4, 0.6, 0.8, 1], StepEnv (kacinci stepte oldugumuzun state'e eklendigi), trajectory h5'inden yuklendigi ilk td :)
#rm test_cartpole_td/test27 -rf
#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model3/reward-24/weights_100000.h5 --batch-size 64 --max-step 100000 --learning-rate 0.005 --target-network-update 500000 --env-model EnvModelCartpole --vmodel CartPoleVNetwork  --load-trajectory test_tensorboard/traj-8_mix.h5 --env-transforms StepEnv --logdir test_cartpole_td/test27

# almost kill the learning rate :)
#rm test_tensorboard/test26 -rf
#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model2/reward-20/weights_100000.h5 --batch-size 64 --max-step 100000 --learning-rate 0.0001 --target-network-update 500000 --env-model EnvModelCartpole --vmodel CartPoleVNetwork  --load-trajectory test_tensorboard/traj.h5 --logdir test_tensorboard/test26

# lower learning rate
#rm test_tensorboard/test25 -rf
#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model2/reward-20/weights_100000.h5 --batch-size 64 --max-step 100000 --learning-rate 0.001 --target-network-update 500000 --env-model EnvModelCartpole --vmodel CartPoleVNetwork  --load-trajectory test_tensorboard/traj.h5 --logdir test_tensorboard/test25

# after multiplying negative rewards with 100
#rm test_tensorboard/test24 -rf
#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model2/reward-20/weights_100000.h5 --batch-size 64 --max-step 100000 --learning-rate 0.005 --target-network-update 500000 --env-model EnvModelCartpole --vmodel CartPoleVNetwork  --load-trajectory test_tensorboard/traj.h5 --logdir test_tensorboard/test24

# after adding done flag
#rm test_tensorboard/test23 -rf
#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model2/reward-20/weights_100000.h5 --batch-size 64 --max-step 100000 --learning-rate 0.005 --target-network-update 500000 --env-model EnvModelCartpole --vmodel CartPoleVNetwork  --load-trajectory test_tensorboard/traj.h5 --logdir test_tensorboard/test23

# same run, with NetworkSaver
#rm test_tensorboard/test22 -rf
#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model/reward-17/weights_500000.h5 --batch-size 64 --max-step 100000 --learning-rate 0.005 --target-network-update 500000 --model CartPoleModel --load-trajectory test_tensorboard/traj.h5 --logdir test_tensorboard/test22

# again without v_model_eval, target-network-update discarted
#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model/reward-17/weights_500000.h5 --batch-size 64 --max-step 100000 --learning-rate 0.005 --target-network-update 500000 --model CartPoleModel --load-trajectory test_tensorboard/traj.h5 --logdir test_tensorboard/test21

# target-network-update = 5000
#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model/reward-17/weights_500000.h5 --batch-size 64 --max-step 100000 --learning-rate 0.005 --target-network-update 5000 --model CartPoleModel --load-trajectory test_tensorboard/traj.h5 --logdir test_tensorboard/test20

# target-network-update = 50
#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model/reward-17/weights_500000.h5 --batch-size 64 --max-step 100000 --learning-rate 0.005 --target-network-update 50 --model CartPoleModel --load-trajectory test_tensorboard/traj.h5 --logdir test_tensorboard/test19


# target-network-update = 10, same experiment with previous
#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model/reward-17/weights_500000.h5 --batch-size 64 --max-step 100000 --learning-rate 0.005 --target-network-update 10 --model CartPoleModel --load-trajectory test_tensorboard/traj.h5 --logdir test_tensorboard/test18

# target-network-update = 10
#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model/reward-17/weights_500000.h5 --batch-size 64 --max-step 100000 --learning-rate 0.005 --target-network-update 10 --model CartPoleModel --load-trajectory test_tensorboard/traj.h5 --logdir test_tensorboard/test17

# target-network-update = 1000
#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model/reward-17/weights_500000.h5 --batch-size 64 --max-step 100000 --learning-rate 0.005 --target-network-update 1000 --model CartPoleModel --load-trajectory test_tensorboard/traj.h5 --logdir test_tensorboard/test16

# target-network-update = 100
#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model/reward-17/weights_500000.h5 --batch-size 64 --max-step 100000 --learning-rate 0.005 --target-network-update 100 --model CartPoleModel --load-trajectory test_tensorboard/traj.h5 --logdir test_tensorboard/test15

# samples are now from DQN trajectory loader
#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model/reward-17/weights_500000.h5 --batch-size 64 --max-step 100000 --learning-rate 0.005 --target-network-update 10000 --model CartPoleModel --load-trajectory test_tensorboard/traj.h5 --logdir test_tensorboard/test14

# samples are now from DQN
#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model/reward-17/weights_500000.h5 --batch-size 64 --max-step 100000 --learning-rate 0.005 --target-network-update 10000 --model CartPoleModel --replay-start-size 64 --replay-buffer-size 10000 --batch-size 64 --update-frequency 1 --load-weightfile test_cartpole2/dqn-7/weights_50000.h5 --test-epsilon 0.6  --load-trajectory test_tensorboard/traj.h5 --logdir test_tensorboard/test13

# target-network-update = 10000
#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model/reward-17/weights_500000.h5 --batch-size 64 --max-step 100000 --learning-rate 0.005 --target-network-update 10000 --logdir test_tensorboard/test12

# target-network-update = 1
#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model/reward-17/weights_500000.h5 --batch-size 64 --max-step 100000 --learning-rate 0.005 --target-network-update 1 --logdir test_tensorboard/test11

# target-network-update = 100
#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model/reward-17/weights_500000.h5 --batch-size 64 --max-step 100000 --learning-rate 0.005 --target-network-update 100 --logdir test_tensorboard/test10

# vmodel is now in VNetwork class. v_model_eval is used in next v estimation and updated at each 1000th batch
#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model/reward-17/weights_500000.h5 --batch-size 64 --max-step 100000 --learning-rate 0.005 --logdir test_tensorboard/test9

# default initialization
#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model/reward-17/weights_500000.h5 --batch-size 64 --max-step 100000 --learning-rate 0.005 --logdir test_tensorboard/test8

#RMSprop(lr=self.ops.LEARNING_RATE)
#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model/reward-17/weights_500000.h5 --batch-size 64 --max-step 100000 --learning-rate 0.005 --logdir test_tensorboard/test7

# next_v.trainable = False
#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model/reward-17/weights_500000.h5 --batch-size 64 --max-step 100000 --learning-rate 0.005 --logdir test_tensorboard/test6

# v hesabinda yanlislikla relu kalmis. v = Dense(1, kernel_initializer='he_uniform')(x) #activation="relu", 
#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model/reward-17/weights_500000.h5 --batch-size 64 --max-step 100000 --learning-rate 0.005 --logdir test_tensorboard/test5

# enable trainable of V value within max, v:24x24x1 instead of 24x1
#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model/reward-17/weights_500000.h5 --batch-size 64 --max-step 100000 --learning-rate 0.005 --logdir test_cartpole_td/test-11

# disable trainable of V value within max *************
#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model/reward-17/weights_500000.h5 --batch-size 64 --max-step 100000 --learning-rate 0.005 --logdir test_cartpole_td/test-10

#RMSprop(lr=self.ops.LEARNING_RATE, rho=0.95, epsilon=None, decay=0.0)  *************
#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model/reward-17/weights_500000.h5 --batch-size 64 --max-step 100000 --learning-rate 0.005 --logdir test_cartpole_td/test-9

#RMSprop(lr=self.ops.LEARNING_RATE, rho=0.9, epsilon=None, decay=0.0)  *************
#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model/reward-17/weights_500000.h5 --batch-size 64 --max-step 100000 --learning-rate 0.005 --logdir test_cartpole_td/test-8

#RMSprop(lr=self.ops.LEARNING_RATE, rho=0.9, epsilon=None, decay=0.0)  *************
#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model/reward-17/weights_500000.h5 --batch-size 64 --max-step 100000 --learning-rate 0.00025 --logdir test_cartpole_td/test-7

#RMSprop(lr=self.ops.LEARNING_RATE, rho=0.9, epsilon=None, decay=0.0)  ************* BURADA BIR KARISIKLIK OLMUS, HALA ADAM KULLANILIYOR GALIBA, SONRAKI RUNLAR DAHIL ************
#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model/reward-17/weights_500000.h5 --batch-size 64 --max-step 100000 --learning-rate 0.001 --logdir test_cartpole_td/test-6

#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model/reward-17/weights_500000.h5 --batch-size 64 --max-step 100000 --learning-rate 0.00001 --logdir test_cartpole_td/test-5

#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model/reward-17/weights_500000.h5 --batch-size 64 --max-step 100000 --learning-rate 0.001 --logdir test_cartpole_td/test-4 

# freeze env_model
#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model/reward-17/weights_500000.h5 --batch-size 64 --max-step 10000 --logdir test_cartpole_td/test-3

#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model/reward-17/weights_500000.h5 --batch-size 64 --max-step 10000 --logdir test_cartpole_td/test-2

#python -B oo_td.py CartPole-v1 --env-weightfile test_cartpole_model/reward-17/weights_500000.h5 --batch-size 64 --max-step 100 --logdir test_cartpole_td/test-1
