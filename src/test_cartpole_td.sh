python -B oo_td_test.py CartPole-v1 --env-model EnvModelCartPoleManual --max-episode 1 --vmodel CartPoleVNetwork --load-weightfile  figures/algo_convergence_cartpole/td_realtime/train-0/weights_ 1000 1000 50001 --env-transform Penalizer --monitor-dir figures/algo_convergence_cartpole/td_realtime/train-0/monitor

#python -B oo_td_test.py CartPole-v1 --env-model EnvModelCartPoleManual --max-episode 1 --vmodel CartPoleVNetwork --load-weightfile  figures/algo_convergence_cartpole/td/train-1/weights_ 1000 1000 50001 --env-transform Penalizer --monitor-dir figures/algo_convergence_cartpole/td/train-1/monitor

#python -B oo_td_test.py CartPole-v1 --env-model EnvModelCartPoleManual --max-episode 1 --vmodel CartPoleVNetwork --load-weightfile  figures/algo_convergence_cartpole/td/train-0/weights_ 1000 1000 50001 --env-transform Penalizer --monitor-dir figures/algo_convergence_cartpole/td/train-0/monitor

# bir onceki ile ayni. VAgent'ta sadece next state'e bakiliyordu, artik reward + next_state'e bakiliyor 
#rm test_cartpole_v/test24 -rf
#python -B oo_td_test.py CartPole-v1 --env-model EnvModelCartpole --env-weightfile test_cartpole_model3/reward-25/weights_100000.h5 --max-episode 5 --vmodel CartPoleVNetwork --load-weightfile  test_cartpole_td/test39/weights_ 1 1 1000 --env-transform Penalizer --logdir test_cartpole_v/test24

# td-test30, v-test14 tekrari x 500 step
# use different trajectory sample epsilon = [0.2, 0.4, 0.6] trajectory_utils/TrajectoryLoader'da manuel kod degistirildi
#rm test_cartpole_v/test23 -rf
#python -B oo_td_test.py CartPole-v1 --env-model EnvModelCartpole --env-weightfile test_cartpole_model3/reward-25/weights_100000.h5 --max-episode 5 --vmodel CartPoleVNetwork --load-weightfile  test_cartpole_td/test39/weights_ 1 1 1000 --env-transform Penalizer --logdir test_cartpole_v/test23

# td-test30, v-test14 tekrari x 1000 step
# use different trajectory sample epsilon = [0.2, 0.4, 0.6] trajectory_utils/TrajectoryLoader'da manuel kod degistirildi
#rm test_cartpole_v/test22 -rf
#python -B oo_td_test.py CartPole-v1 --env-model EnvModelCartpole --env-weightfile test_cartpole_model3/reward-25/weights_100000.h5 --max-episode 5 --vmodel CartPoleVNetwork --load-weightfile  test_cartpole_td/test38/weights_ 10 10 1000 --env-transform Penalizer --logdir test_cartpole_v/test22

# td-test30, v-test14 tekrari x 100 step
#rm test_cartpole_v/test21 -rf
#python -B oo_td_test.py CartPole-v1 --env-model EnvModelCartpole --env-weightfile test_cartpole_model3/reward-25/weights_100000.h5 --max-episode 5 --vmodel CartPoleVNetwork --load-weightfile  test_cartpole_td/test36/weights_ 1 1 100 --env-transform Penalizer --logdir test_cartpole_v/test21

# td-test30, v-test14 tekrari x 1000 step
#rm test_cartpole_v/test20 -rf
#python -B oo_td_test.py CartPole-v1 --env-model EnvModelCartpole --env-weightfile test_cartpole_model3/reward-25/weights_100000.h5 --max-episode 5 --vmodel CartPoleVNetwork --load-weightfile  test_cartpole_td/test37/weights_ 10 10 1000 --env-transform Penalizer --logdir test_cartpole_v/test20

# bu en iyiye yakindi sanirim
#rm test_cartpole_v/test19 -rf
#python -B oo_td_test.py CartPole-v1 --env-model EnvModelCartpole --env-weightfile test_cartpole_model3/reward-25/weights_100000.h5 --max-episode 5 --vmodel CartPoleVNetwork --load-weightfile  test_cartpole_td/test35/weights_ 1000 1000 100000 --env-transform Penalizer --logdir test_cartpole_v/test19

# train_cartpole_sim_dqn.sh ile test ettigimde en iyi sonucu veren env'in testi - BUNUN SONUCLARINA GUVENME
#rm test_cartpole_v/test18 -rf
#python -B oo_td_test.py CartPole-v1 --env-model EnvModelCartpole --env-weightfile test_cartpole_model3/reward-27/weights_300000.h5 --max-episode 5 --vmodel CartPoleVNetwork --load-weightfile  test_cartpole_td/test34/weights_ 1000 1000 100000 --env-transform Penalizer --logdir test_cartpole_v/test18

# simdilik gecici testler 
#rm test_cartpole_v/test17 -rf
#python -B oo_td_test.py CartPole-v1 --env-model EnvModelCartpole --env-weightfile test_cartpole_model3/reward-25/weights_100000.h5 --max-episode 5 --vmodel CartPoleVNetwork --load-weightfile  test_cartpole_td/test33/weights_ 1000 1000 250000 --env-transform Penalizer --logdir test_cartpole_v/test17

# ADAM'dan RMSprop'a gecildi
#rm test_cartpole_v/test16 -rf
#python -B oo_td_test.py CartPole-v1 --env-model EnvModelCartpole --env-weightfile test_cartpole_model3/reward-25/weights_100000.h5 --max-episode 5 --vmodel CartPoleVNetwork --load-weightfile  test_cartpole_td/test32/weights_ 1000 1000 250000 --env-transform Penalizer --logdir test_cartpole_v/test16

# learning rate dusuruldu: 0.0001 
#rm test_cartpole_v/test15 -rf
#python -B oo_td_test.py CartPole-v1 --env-model EnvModelCartpole --env-weightfile test_cartpole_model3/reward-25/weights_100000.h5 --max-episode 5 --vmodel CartPoleVNetwork --load-weightfile  test_cartpole_td/test31/weights_ 1000 1000 250000 --env-transform Penalizer --logdir test_cartpole_v/test15

# learning rate dusuruldu: 0.001
#rm test_cartpole_v/test14 -rf
#python -B oo_td_test.py CartPole-v1 --env-model EnvModelCartpole --env-weightfile test_cartpole_model3/reward-25/weights_100000.h5 --max-episode 5 --vmodel CartPoleVNetwork --load-weightfile  test_cartpole_td/test30/weights_ 1000 1000 250000 --env-transform Penalizer --logdir test_cartpole_v/test14

# bir onceki ile ayni sadece daha cok network test edildi
#rm test_cartpole_v/test13 -rf
#python -B oo_td_test.py CartPole-v1 --env-model EnvModelCartpole --env-weightfile test_cartpole_model3/reward-25/weights_100000.h5 --max-episode 5 --vmodel CartPoleVNetwork --load-weightfile  test_cartpole_td/test29/weights_ 1000 1000 250000 --env-transform Penalizer --logdir test_cartpole_v/test13

# test the networks obtained during td training on real env
#rm test_cartpole_v/test12 -rf
#python -B oo_td_test.py CartPole-v1 --env-model EnvModelCartpole --env-weightfile test_cartpole_model3/reward-25/weights_100000.h5 --max-step 2000 --vmodel CartPoleVNetwork --load-weightfile  test_cartpole_td/test29/weights_ 10000 5000 250000 --env-transform Penalizer --logdir test_cartpole_v/test12

# bironceki ile ayni sadece, v'nin birazcik daha iyi versiyonu
#rm test_cartpole_v/test11 -rf
#python -B oo_td_test.py CartPole-v1 --env-model EnvModelCartpole --env-weightfile test_cartpole_model3/reward-25/weights_100000.h5 --max-step 100000 --vmodel CartPoleVNetwork --load-weightfile  test_cartpole_td/test28/weights_80000.h5 --env-transform Penalizer --logdir test_cartpole_v/test11

# bironceki ile ayni sadece, v'nin biraz daha iyi versiyonu
#rm test_cartpole_v/test10 -rf
#python -B oo_td_test.py CartPole-v1 --env-model EnvModelCartpole --env-weightfile test_cartpole_model3/reward-25/weights_100000.h5 --max-step 100000 --vmodel CartPoleVNetwork --load-weightfile  test_cartpole_td/test28/weights_50000.h5 --env-transform Penalizer --logdir test_cartpole_v/test10

# bironceki ile ayni sadece, v'nin daha kotu hali ile test ediyorum
#rm test_cartpole_v/test9 -rf
#python -B oo_td_test.py CartPole-v1 --env-model EnvModelCartpole --env-weightfile test_cartpole_model3/reward-25/weights_100000.h5 --max-step 100000 --vmodel CartPoleVNetwork --load-weightfile  test_cartpole_td/test28/weights_10000.h5 --env-transform Penalizer --logdir test_cartpole_v/test9

# AND SHOW TIME, ASAGIDAKI CALISTI
#rm test_cartpole_v/test8 -rf
#python -B oo_td_test.py CartPole-v1 --env-model EnvModelCartpole --env-weightfile test_cartpole_model3/reward-25/weights_100000.h5 --max-step 100000 --vmodel CartPoleVNetwork --load-weightfile  test_cartpole_td/test28/weights_100000.h5 --env-transform Penalizer --logdir test_cartpole_v/test8

# StepEnv eklendikten ve multiple greedy dqn ile train edildikten sonra
#rm test_cartpole_v/test7 -rf
#python -B oo_td_test.py CartPole-v1 --env-model EnvModelCartpole --env-weightfile test_cartpole_model3/reward-24/weights_100000.h5 --max-step 100000 --vmodel CartPoleVNetwork --load-weightfile  test_cartpole_td/test27/weights_15000.h5 --env-transform Penalizer StepEnv --logdir test_cartpole_v/test7

# kill the learning rate :)
#rm test_cartpole_v/test6 -rf
#python -B oo_td_test.py CartPole-v1 --env-model EnvModelCartpole --env-weightfile test_cartpole_model2/reward-20/weights_100000.h5 --max-step 100000 --vmodel CartPoleVNetwork --load-weightfile  test_tensorboard/test26/weights_100000.h5 --env-transform Penalizer --logdir test_cartpole_v/test6 

# lower down learning rate
#rm test_cartpole_v/test5 -rf
#python -B oo_td_test.py CartPole-v1 --env-model EnvModelCartpole --env-weightfile test_cartpole_model2/reward-20/weights_100000.h5 --max-step 100000 --vmodel CartPoleVNetwork --load-weightfile  test_tensorboard/test25/weights_30000.h5 --env-transform Penalizer --logdir test_cartpole_v/test5 

# same with previous, with different saved network
#rm test_cartpole_v/test4 -rf
#python -B oo_td_test.py CartPole-v1 --env-model EnvModelCartpole --env-weightfile test_cartpole_model2/reward-20/weights_100000.h5 --max-step 100000 --vmodel CartPoleVNetwork --load-weightfile  test_tensorboard/test24/weights_60000.h5 --env-transform Penalizer --logdir test_cartpole_v/test4 

# after multiplying negative rewards with 100
#rm test_cartpole_v/test3 -rf
#python -B oo_td_test.py CartPole-v1 --env-model EnvModelCartpole --env-weightfile test_cartpole_model2/reward-20/weights_100000.h5 --max-step 100000 --vmodel CartPoleVNetwork --load-weightfile  test_tensorboard/test24/weights_100000.h5 --env-transform Penalizer --logdir test_cartpole_v/test3 

# after done fix
#rm test_cartpole_v/test2 -rf
#python -B oo_td_test.py CartPole-v1 --env-model EnvModelCartpole --env-weightfile test_cartpole_model2/reward-20/weights_100000.h5 --max-step 100000 --vmodel CartPoleVNetwork --load-weightfile  test_tensorboard/test23/weights_100000.h5 --env-transform Penalizer --logdir test_cartpole_v/test2 

# first run
#rm test_cartpole_v/test1 -rf
#python -B oo_td_test.py CartPole-v1 --env-model EnvModelCartpole --env-weightfile test_cartpole_model/reward-17/weights_500000.h5 --max-step 100000 --vmodel CartPoleVNetwork --load-weightfile  test_tensorboard/test22/weights_10000.h5 --env-transform Penalizer --logdir test_cartpole_v/test1 
