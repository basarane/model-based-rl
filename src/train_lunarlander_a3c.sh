# a3c-12'un tekrari, save'deki problem cozulmus hali
rm test_lunarlander_a3c/a3c-18 -rf
python -B oo_a3c.py LunarLander-v2 --model LunarLanderModel --learning-rate 0.00025 --target-network-update 100 --egreedy-props 5 5 5 --egreedy-final 0.01 0.1 0.5 --egreedy-final-step 30000 30000 30000 --egreedy-decay 1 1 1 --max-step 1000000 --nstep 5 --thread-count 16 --save-interval 1000 --logdir test_lunarlander_a3c/a3c-18 &

# a3c-15'un tekrari, host restart oldu
#rm test_lunarlander_a3c/a3c-17 -rf
#python -B oo_a3c.py LunarLander-v2 --model LunarLanderModel --learning-rate 0.00025 --target-network-update 5000 --egreedy-props 5 5 5 --egreedy-final 0.01 0.1 0.5 --egreedy-final-step 30000 30000 30000 --egreedy-decay 1 1 1 --max-step 1000000 --nstep 5 --thread-count 16 --save-interval 1000 --logdir test_lunarlander_a3c/a3c-17 &

# a3c-14'un tekrari, host restart oldu
#rm test_lunarlander_a3c/a3c-16 -rf
#python -B oo_a3c.py LunarLander-v2 --model LunarLanderModel --learning-rate 0.00025 --target-network-update 1000 --egreedy-props 5 5 5 --egreedy-final 0.01 0.1 0.5 --egreedy-final-step 30000 30000 30000 --egreedy-decay 1 1 1 --max-step 1000000 --nstep 5 --thread-count 16 --save-interval 1000 --logdir test_lunarlander_a3c/a3c-16 &

# increase target network update to 5000
#rm test_lunarlander_a3c/a3c-15 -rf
#python -B oo_a3c.py LunarLander-v2 --model LunarLanderModel --learning-rate 0.00025 --target-network-update 5000 --egreedy-props 5 5 5 --egreedy-final 0.01 0.1 0.5 --egreedy-final-step 30000 30000 30000 --egreedy-decay 1 1 1 --max-step 1000000 --nstep 5 --thread-count 16 --save-interval 1000 --logdir test_lunarlander_a3c/a3c-15 &

# increase target network update to 1000
#rm test_lunarlander_a3c/a3c-14 -rf
#python -B oo_a3c.py LunarLander-v2 --model LunarLanderModel --learning-rate 0.00025 --target-network-update 1000 --egreedy-props 5 5 5 --egreedy-final 0.01 0.1 0.5 --egreedy-final-step 30000 30000 30000 --egreedy-decay 1 1 1 --max-step 1000000 --nstep 5 --thread-count 16 --save-interval 1000 --logdir test_lunarlander_a3c/a3c-14 &

# save network and target eval network copy is put into the tLock (only 5 networks was saved instead of 1000)
#rm test_lunarlander_a3c/a3c-13 -rf
#python -B oo_a3c.py LunarLander-v2 --model LunarLanderModel --learning-rate 0.00025 --target-network-update 200 --egreedy-props 5 5 5 --egreedy-final 0.01 0.1 0.5 --egreedy-final-step 30000 30000 30000 --egreedy-decay 1 1 1 --max-step 1000000 --nstep 5 --thread-count 16 --save-interval 1000 --logdir test_lunarlander_a3c/a3c-13 &

# same with previous with save-interval
#rm test_lunarlander_a3c/a3c-12 -rf
#python -B oo_a3c.py LunarLander-v2 --model LunarLanderModel --learning-rate 0.00025 --target-network-update 100 --egreedy-props 5 5 5 --egreedy-final 0.01 0.1 0.5 --egreedy-final-step 30000 30000 30000 --egreedy-decay 1 1 1 --max-step 1000000 --nstep 5 --thread-count 16 --save-interval 1000 --logdir test_lunarlander_a3c/a3c-12 &

# target network update is lowered, egreedy-final-step is 30000 x 16 ~ 480000 across all threads now
#rm test_lunarlander_a3c/a3c-11 -rf
#python -B oo_a3c.py LunarLander-v2 --model LunarLanderModel --learning-rate 0.00025 --target-network-update 200 --egreedy-props 3 5 7 --egreedy-final 0.01 0.1 0.5 --egreedy-final-step 30000 30000 30000 --egreedy-decay 1 1 1 --max-step 1000000 --nstep 5 --thread-count 16 --logdir test_lunarlander_a3c/a3c-11 &

# unused parameters removed, parallel tests on (target-network-update, egreedy-props, egreedy-final-step)
#rm test_lunarlander_a3c/a3c-10 -rf
#python -B oo_a3c.py LunarLander-v2 --model LunarLanderModel --learning-rate 0.00025 --target-network-update 1000 --egreedy-props 3 5 7 --egreedy-final 0.01 0.1 0.5 --egreedy-final-step 60000 60000 60000 --egreedy-decay 1 1 1 --max-step 1000000 --nstep 5 --thread-count 16 --logdir test_lunarlander_a3c/a3c-10 &
#
#rm test_lunarlander_a3c/a3c-9 -rf
#python -B oo_a3c.py LunarLander-v2 --model LunarLanderModel --learning-rate 0.00025 --target-network-update 3000 --egreedy-props 3 5 7 --egreedy-final 0.01 0.1 0.5 --egreedy-final-step 60000 60000 60000 --egreedy-decay 1 1 1 --max-step 1000000 --nstep 5 --thread-count 16 --logdir test_lunarlander_a3c/a3c-9 &
#
#rm test_lunarlander_a3c/a3c-8 -rf
#python -B oo_a3c.py LunarLander-v2 --model LunarLanderModel --learning-rate 0.00025 --target-network-update 5000 --egreedy-props 3 5 7 --egreedy-final 0.01 0.1 0.5 --egreedy-final-step 30000 30000 30000 --egreedy-decay 1 1 1 --max-step 1000000 --nstep 5 --thread-count 16 --logdir test_lunarlander_a3c/a3c-8 &
#
#rm test_lunarlander_a3c/a3c-7 -rf
#python -B oo_a3c.py LunarLander-v2 --model LunarLanderModel --learning-rate 0.00025 --target-network-update 5000 --egreedy-props 1 5 9 --egreedy-final 0.01 0.1 0.5 --egreedy-final-step 30000 30000 30000 --egreedy-decay 1 1 1 --max-step 1000000 --nstep 5 --thread-count 16 --logdir test_lunarlander_a3c/a3c-7 &

#16 thread, egreedy_props are now also support int (i.e egreedy_counts)
#rm test_lunarlander_a3c/a3c-6 -rf
#python -B oo_a3c.py LunarLander-v2 --model LunarLanderModel --learning-rate 0.00025 --target-network-update 5000 --egreedy-props 6 5 4 --egreedy-final 0.01 0.1 0.5 --egreedy-final-step 30000 30000 30000 --egreedy-decay 1 1 1 --replay-start-size 5000 --replay-buffer-size 50000 --batch-size 64 --update-frequency 1 --max-step 1000000 --nstep 5 --thread-count 16 --logdir test_lunarlander_a3c/a3c-6

# greedy-final-step is now per thread. For 8 threads, the mean max step per thread was 130k. So around 1/4 30k is selected for egreedy final step
#rm test_lunarlander_a3c/a3c-5 -rf
#python -B oo_a3c.py LunarLander-v2 --model LunarLanderModel --learning-rate 0.00025 --target-network-update 5000 --egreedy-props 0.4 0.3 0.3 --egreedy-final 0.01 0.1 0.5 --egreedy-final-step 30000 30000 30000 --egreedy-decay 1 1 1 --replay-start-size 5000 --replay-buffer-size 50000 --batch-size 64 --update-frequency 1 --max-step 1000000 --nstep 5 --thread-count 8 --logdir test_lunarlander_a3c/a3c-5

# RMSprop + mse
#rm test_lunarlander_a3c/a3c-4 -rf
#python -B oo_a3c.py LunarLander-v2 --model LunarLanderModel --learning-rate 0.00025 --target-network-update 5000 --egreedy-props 0.4 0.3 0.3 --egreedy-final 0.01 0.1 0.5 --egreedy-final-step 250000 250000 250000 --egreedy-decay 1 1 1 --replay-start-size 5000 --replay-buffer-size 50000 --batch-size 64 --update-frequency 1 --max-step 1000000 --nstep 5 --thread-count 8 --logdir test_lunarlander_a3c/a3c-4

# RMSprop + huber_loss
#rm test_lunarlander_a3c/a3c-3 -rf
#python -B oo_a3c.py LunarLander-v2 --model LunarLanderModel --learning-rate 0.00025 --target-network-update 5000 --egreedy-props 0.4 0.3 0.3 --egreedy-final 0.01 0.1 0.5 --egreedy-final-step 250000 250000 250000 --egreedy-decay 1 1 1 --replay-start-size 5000 --replay-buffer-size 50000 --batch-size 64 --update-frequency 1 --max-step 1000000 --nstep 5 --thread-count 8 --logdir test_lunarlander_a3c/a3c-3

# Adam + huber_loss
#rm test_lunarlander_a3c/a3c-2 -rf
#python -B oo_a3c.py LunarLander-v2 --model LunarLanderModel --learning-rate 0.00025 --target-network-update 5000 --egreedy-props 0.4 0.3 0.3 --egreedy-final 0.01 0.1 0.5 --egreedy-final-step 250000 250000 250000 --egreedy-decay 1 1 1 --replay-start-size 5000 --replay-buffer-size 50000 --batch-size 64 --update-frequency 1 --max-step 1000000 --nstep 5 --thread-count 8 --logdir test_lunarlander_a3c/a3c-2

# first test: RMSProp + huber_loss_mse
#rm test_lunarlander_a3c/a3c-1 -rf
#python -B oo_a3c.py LunarLander-v2 --model LunarLanderModel --learning-rate 0.00025 --target-network-update 5000 --egreedy-props 0.4 0.3 0.3 --egreedy-final 0.01 0.1 0.5 --egreedy-final-step 250000 250000 250000 --egreedy-decay 1 1 1 --replay-start-size 5000 --replay-buffer-size 50000 --batch-size 64 --update-frequency 1 --max-step 1000000 --nstep 5 --thread-count 8 --logdir test_lunarlander_a3c/a3c-1
