#VAgent'ta sadece next state'e bakiliyordu, artik reward + next_state'e bakiliyor {prediction[I] = (r*100 if r<0 else r) + (1-done)0.99*self.v_model.v_value(next_obs[I])[0]}
rm test_cartpole_td_realtime/test-3 -rf
python -B oo_td_realtime.py CartPole-v1 --env-weightfile test_cartpole_model3/reward-25/weights_100000.h5 --batch-size 64 --max-step 100000 --learning-rate 0.001 --target-network-update 500000 --env-model EnvModelCartpole --vmodel CartPoleVNetwork --save-freq 1000 --replay-start-size 1000 --replay-buffer-size 10000 --env-transforms Penalizer --logdir test_cartpole_td_realtime/test-3

# sadece maxstep arttirildi
#rm test_cartpole_td_realtime/test-2 -rf
#python -B oo_td_realtime.py CartPole-v1 --env-weightfile test_cartpole_model3/reward-25/weights_100000.h5 --batch-size 64 --max-step 100000 --learning-rate 0.001 --target-network-update 500000 --env-model EnvModelCartpole --vmodel CartPoleVNetwork --save-freq 1000 --replay-start-size 1000 --replay-buffer-size 10000 --env-transforms Penalizer --logdir test_cartpole_td_realtime/test-2

# ilk test
#mkdir test_cartpole_td_realtime
#rm test_cartpole_td_realtime/test-1 -rf
#python -B oo_td_realtime.py CartPole-v1 --env-weightfile test_cartpole_model3/reward-25/weights_100000.h5 --batch-size 64 --max-step 50000 --learning-rate 0.001 --target-network-update 500000 --env-model EnvModelCartpole --vmodel CartPoleVNetwork --save-freq 1000 --replay-start-size 1000 --replay-buffer-size 10000 --env-transforms Penalizer --logdir test_cartpole_td_realtime/test-1

