#  2 hidden layers 5 more times in parallel
rm test_line_td_realtime/test-41 -rf
python -B oo_td_realtime.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork --save-freq 10 --replay-start-size 64 --replay-buffer-size 10000 --egreedy-final 0.1 --egreedy-final-step 10000 --egreedy-decay 1 --update-frequency 1 --logdir test_line_td_realtime/test-41 &

rm test_line_td_realtime/test-40 -rf
python -B oo_td_realtime.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork --save-freq 10 --replay-start-size 64 --replay-buffer-size 10000 --egreedy-final 0.1 --egreedy-final-step 10000 --egreedy-decay 1 --update-frequency 1 --logdir test_line_td_realtime/test-40 & 

rm test_line_td_realtime/test-39 -rf
python -B oo_td_realtime.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork --save-freq 10 --replay-start-size 64 --replay-buffer-size 10000 --egreedy-final 0.1 --egreedy-final-step 10000 --egreedy-decay 1 --update-frequency 1 --logdir test_line_td_realtime/test-39 &

rm test_line_td_realtime/test-38 -rf
python -B oo_td_realtime.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork --save-freq 10 --replay-start-size 64 --replay-buffer-size 10000 --egreedy-final 0.1 --egreedy-final-step 10000 --egreedy-decay 1 --update-frequency 1 --logdir test_line_td_realtime/test-38 &

rm test_line_td_realtime/test-37 -rf
python -B oo_td_realtime.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork --save-freq 10 --replay-start-size 64 --replay-buffer-size 10000 --egreedy-final 0.1 --egreedy-final-step 10000 --egreedy-decay 1 --update-frequency 1 --logdir test_line_td_realtime/test-37 & 

# 2 hidden layers (ReLU) 10x10x1 
#rm test_line_td_realtime/test-36 -rf
#python -B oo_td_realtime.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork --save-freq 10 --replay-start-size 64 --replay-buffer-size 10000 --egreedy-final 0.1 --egreedy-final-step 10000 --egreedy-decay 1 --update-frequency 1 --logdir test_line_td_realtime/test-36 

# linear e-greedy 5 more times in parallel
#rm test_line_td_realtime/test-35 -rf
#python -B oo_td_realtime.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork --save-freq 10 --replay-start-size 64 --replay-buffer-size 10000 --egreedy-final 0.1 --egreedy-final-step 10000 --egreedy-decay 1 --update-frequency 1 --logdir test_line_td_realtime/test-35 &
#
#rm test_line_td_realtime/test-34 -rf
#python -B oo_td_realtime.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork --save-freq 10 --replay-start-size 64 --replay-buffer-size 10000 --egreedy-final 0.1 --egreedy-final-step 10000 --egreedy-decay 1 --update-frequency 1 --logdir test_line_td_realtime/test-34 &
#
#rm test_line_td_realtime/test-33 -rf
#python -B oo_td_realtime.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork --save-freq 10 --replay-start-size 64 --replay-buffer-size 10000 --egreedy-final 0.1 --egreedy-final-step 10000 --egreedy-decay 1 --update-frequency 1 --logdir test_line_td_realtime/test-33 &
#
#rm test_line_td_realtime/test-32 -rf
#python -B oo_td_realtime.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork --save-freq 10 --replay-start-size 64 --replay-buffer-size 10000 --egreedy-final 0.1 --egreedy-final-step 10000 --egreedy-decay 1 --update-frequency 1 --logdir test_line_td_realtime/test-32 &
#
#rm test_line_td_realtime/test-31 -rf
#python -B oo_td_realtime.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork --save-freq 10 --replay-start-size 64 --replay-buffer-size 10000 --egreedy-final 0.1 --egreedy-final-step 10000 --egreedy-decay 1 --update-frequency 1 --logdir test_line_td_realtime/test-31 &

# linear epsilon greedy test
#rm test_line_td_realtime/test-30 -rf
#python -B oo_td_realtime.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork --save-freq 10 --replay-start-size 64 --replay-buffer-size 10000 --egreedy-final 0.1 --egreedy-final-step 10000 --egreedy-decay 1 --update-frequency 1 --logdir test_line_td_realtime/test-30

# one more time (previous one does not converged)
#rm test_line_td_realtime/test-29 -rf
#python -B oo_td_realtime.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork --save-freq 10 --replay-start-size 64 --replay-buffer-size 10000 --egreedy-final 0.01 --egreedy-decay 0.99 --update-frequency 1 --logdir test_line_td_realtime/test-29

# increase save-freq
#rm test_line_td_realtime/test-28 -rf
#python -B oo_td_realtime.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork --save-freq 10 --replay-start-size 64 --replay-buffer-size 10000 --egreedy-final 0.01 --egreedy-decay 0.99 --update-frequency 1 --logdir test_line_td_realtime/test-28

# one more time 
#rm test_line_td_realtime/test-27 -rf
#python -B oo_td_realtime.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork --save-freq 1000 --replay-start-size 64 --replay-buffer-size 10000 --egreedy-final 0.01 --egreedy-decay 0.99 --update-frequency 1 --logdir test_line_td_realtime/test-27

# same again 
#rm test_line_td_realtime/test-26 -rf
#python -B oo_td_realtime.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork --save-freq 1000 --replay-start-size 64 --replay-buffer-size 10000 --egreedy-final 0.01 --egreedy-decay 0.99 --update-frequency 1 --logdir test_line_td_realtime/test-26

# same with previous
#rm test_line_td_realtime/test-25 -rf
#python -B oo_td_realtime.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork --save-freq 1000 --replay-start-size 64 --replay-buffer-size 10000 --egreedy-final 0.01 --egreedy-decay 0.99 --update-frequency 1 --logdir test_line_td_realtime/test-25

# 		my_optimizer = DqnRMSprop(lr=self.ops.LEARNING_RATE, rho1=0.95, rho2=0.95, epsilon=0.01, print_layer=-1)
#		model.compile(optimizer=my_optimizer,loss=huber_loss) #
#rm test_line_td_realtime/test-24 -rf
#python -B oo_td_realtime.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork --save-freq 1000 --replay-start-size 64 --replay-buffer-size 10000 --egreedy-final 0.01 --egreedy-decay 0.99 --update-frequency 1 --logdir test_line_td_realtime/test-24

# refactor test, same parameters
#rm test_line_td_realtime/test-23 -rf
#python -B oo_td_realtime.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork --save-freq 1000 --replay-start-size 64 --replay-buffer-size 10000 --egreedy-final 0.01 --egreedy-decay 0.99 --update-frequency 1 --logdir test_line_td_realtime/test-23

# 5 more times :)) up to now, 3 out of 8 times the same setting converged faster than DQN, it fails to converge 5/8
# final convergence 5 / 13
#rm test_line_td_realtime/test-22 -rf
#python -B oo_td_realtime.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork --save-freq 1000 --replay-start-size 64 --replay-buffer-size 10000 --egreedy-final 0.01 --egreedy-decay 0.99 --update-frequency 1 --logdir test_line_td_realtime/test-22
#
#rm test_line_td_realtime/test-21 -rf
#python -B oo_td_realtime.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork --save-freq 1000 --replay-start-size 64 --replay-buffer-size 10000 --egreedy-final 0.01 --egreedy-decay 0.99 --update-frequency 1 --logdir test_line_td_realtime/test-21
#
#rm test_line_td_realtime/test-20 -rf
#python -B oo_td_realtime.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork --save-freq 1000 --replay-start-size 64 --replay-buffer-size 10000 --egreedy-final 0.01 --egreedy-decay 0.99 --update-frequency 1 --logdir test_line_td_realtime/test-20
#
#rm test_line_td_realtime/test-19 -rf
#python -B oo_td_realtime.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork --save-freq 1000 --replay-start-size 64 --replay-buffer-size 10000 --egreedy-final 0.01 --egreedy-decay 0.99 --update-frequency 1 --logdir test_line_td_realtime/test-19
#
#rm test_line_td_realtime/test-18 -rf
#python -B oo_td_realtime.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork --save-freq 1000 --replay-start-size 64 --replay-buffer-size 10000 --egreedy-final 0.01 --egreedy-decay 0.99 --update-frequency 1 --logdir test_line_td_realtime/test-18

# 3 more times (15-16-17)
#rm test_line_td_realtime/test-17 -rf
#python -B oo_td_realtime.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork --save-freq 1000 --replay-start-size 64 --replay-buffer-size 10000 --egreedy-final 0.01 --egreedy-decay 0.99 --update-frequency 1 --logdir test_line_td_realtime/test-17

#rm test_line_td_realtime/test-16 -rf
#python -B oo_td_realtime.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork --save-freq 1000 --replay-start-size 64 --replay-buffer-size 10000 --egreedy-final 0.01 --egreedy-decay 0.99 --update-frequency 1 --logdir test_line_td_realtime/test-16

#rm test_line_td_realtime/test-15 -rf
#python -B oo_td_realtime.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork --save-freq 1000 --replay-start-size 64 --replay-buffer-size 10000 --egreedy-final 0.01 --egreedy-decay 0.99 --update-frequency 1 --logdir test_line_td_realtime/test-15

# last time with same parameters
#rm test_line_td_realtime/test-14 -rf
#python -B oo_td_realtime.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork --save-freq 1000 --replay-start-size 64 --replay-buffer-size 10000 --egreedy-final 0.01 --egreedy-decay 0.99 --update-frequency 1 --logdir test_line_td_realtime/test-14

# one more time
#rm test_line_td_realtime/test-13 -rf
#python -B oo_td_realtime.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork --save-freq 1000 --replay-start-size 64 --replay-buffer-size 10000 --egreedy-final 0.01 --egreedy-decay 0.99 --update-frequency 1 --logdir test_line_td_realtime/test-13

# same run third time
#rm test_line_td_realtime/test-12 -rf
#python -B oo_td_realtime.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork --save-freq 1000 --replay-start-size 64 --replay-buffer-size 10000 --egreedy-final 0.01 --egreedy-decay 0.99 --update-frequency 1 --logdir test_line_td_realtime/test-12

# same run with previous
#rm test_line_td_realtime/test-11 -rf
#python -B oo_td_realtime.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork --save-freq 1000 --replay-start-size 64 --replay-buffer-size 10000 --egreedy-final 0.01 --egreedy-decay 0.99 --update-frequency 1 --logdir test_line_td_realtime/test-11

# en iyi calisan test_line_dqn/linedqn-3 ile ayni parametreler
#rm test_line_td_realtime/test-10 -rf
#python -B oo_td_realtime.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork --save-freq 1000 --replay-start-size 64 --replay-buffer-size 10000 --egreedy-final 0.01 --egreedy-decay 0.99 --update-frequency 1 --logdir test_line_td_realtime/test-10

# v-network: he_uniform 100 hidden unit, mean_absolute_error, Adam - BUNUN SONUCLARI DOGRU DEGILDIR
#rm test_line_td_realtime/test-9 -rf
#python -B oo_td_realtime.py Line --batch-size 64 --max-step 50000 --learning-rate 0.0003 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork --save-freq 1000 --replay-start-size 64 --replay-buffer-size 10000 --egreedy-final 0.01 --egreedy-decay 0.99 --update-frequency 1 --logdir test_line_td_realtime/test-9

# learning rate arttirildi
#rm test_line_td_realtime/test-8 -rf
#python -B oo_td_realtime.py Line --batch-size 64 --max-step 50000 --learning-rate 0.01 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork --save-freq 1000 --replay-start-size 64 --replay-buffer-size 10000 --egreedy-final 0.01 --egreedy-decay 0.99 --logdir test_line_td_realtime/test-8

# model_eval hala yok, ama next_v_tensor[:].trainable = False
#rm test_line_td_realtime/test-7 -rf
#python -B oo_td_realtime.py Line --batch-size 64 --max-step 50000 --learning-rate 0.001 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork --save-freq 1000 --replay-start-size 64 --replay-buffer-size 10000 --egreedy-final 0.01 --egreedy-decay 0.99 --logdir test_line_td_realtime/test-7

# model_eval iptal edildi, ayrica next_v_tensor'undeki v'ler trainable yapildi
#rm test_line_td_realtime/test-6 -rf
#python -B oo_td_realtime.py Line --batch-size 64 --max-step 50000 --learning-rate 0.001 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork --save-freq 1000 --replay-start-size 64 --replay-buffer-size 10000 --egreedy-final 0.01 --egreedy-decay 0.99 --logdir test_line_td_realtime/test-6

# VAgent'ta training sirasinda test icin randomuniform kodu vardi, kaldirildi. Tekrar ReplayBuffer'dan aliyor (cikarilan satir: current_states = np.random.random((64,1))*2-1)
#rm test_line_td_realtime/test-5 -rf
#python -B oo_td_realtime.py Line --batch-size 64 --max-step 50000 --learning-rate 0.001 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork --save-freq 1000 --replay-start-size 64 --replay-buffer-size 10000 --egreedy-final 0.01 --egreedy-decay 0.99 --logdir test_line_td_realtime/test-5

#rm test_line_td_realtime/test-4 -rf
#python -B oo_td_realtime.py Line --batch-size 64 --max-step 50000 --learning-rate 0.001 --target-network-update 30 --env-model EnvModelLineManual --vmodel LineVNetwork --save-freq 1000 --replay-start-size 100 --replay-buffer-size 10000 --egreedy-final 0.01 --egreedy-decay 0.998 --logdir test_line_td_realtime/test-4

# target-network-update aktif edildi - td_error yanlislikla tum matrix'i cikartiyormus o da her seyi 1'e cekiyordu v - est_v' oluyordu, broadcast yapiyordu
#rm test_line_td_realtime/test-3 -rf
#python -B oo_td_realtime.py Line --batch-size 64 --max-step 50000 --learning-rate 0.001 --target-network-update 5000 --env-model EnvModelLineManual --vmodel LineVNetwork --save-freq 1000 --replay-start-size 100 --replay-buffer-size 10000 --egreedy-final 0.01 --egreedy-decay 0.998 --logdir test_line_td_realtime/test-3

# EnvModelLineManual ile ilk test
#rm test_line_td_realtime/test-2 -rf
#python -B oo_td_realtime.py Line --batch-size 64 --max-step 50000 --learning-rate 0.001 --target-network-update 500000 --env-model EnvModelLineManual --vmodel LineVNetwork --save-freq 1000 --replay-start-size 100 --replay-buffer-size 10000 --egreedy-final 0.01 --egreedy-decay 0.998 --logdir test_line_td_realtime/test-2

#ilk test
#mkdir test_line_td_realtime
#rm test_line_td_realtime/test-1 -rf
#python -B oo_td_realtime.py Line --env-weightfile test_line_env/lineenv-1/weights_5000.h5 --batch-size 64 --max-step 50000 --learning-rate 0.001 --target-network-update 500000 --env-model EnvModelLine --vmodel LineVNetwork --save-freq 1000 --replay-start-size 100 --replay-buffer-size 10000 --egreedy-final 0.01 --egreedy-decay 0.998 --logdir test_line_td_realtime/test-1
