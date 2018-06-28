# minor network refactor: ActorCriticModel is now base class
rm test_cartpole_a3c/a3c-20 -rf
python -B oo_a3c.py CartPole-v1 --model CartPoleActorCriticModel --agent ActorCriticAgent --learning-rate 0.001 --target-network-update 20 --egreedy-final 0.01 --egreedy-decay 0.9999 0.999 0.998 --egreedy-props 0.4 0.3 0.3 --replay-start-size 1000 --replay-buffer-size 20000 --batch-size 64 --update-frequency 1 --thread-count 8 --nstep 1000 --max-step 600000 --shared-model True --gpu 1 --logdir test_cartpole_a3c/a3c-20

## run 4 gpu tests on parallel
#rm test_cartpole_a3c/a3c-19 -rf
#python -B oo_a3c.py CartPole-v1 --model ActorCriticModel --agent ActorCriticAgent --learning-rate 0.001 --target-network-update 20 --egreedy-final 0.01 --egreedy-decay 0.9999 0.999 0.998 --egreedy-props 0.4 0.3 0.3 --replay-start-size 1000 --replay-buffer-size 20000 --batch-size 64 --update-frequency 1 --thread-count 8 --nstep 1000 --max-step 600000 --shared-model True --gpu 1 --logdir test_cartpole_a3c/a3c-19 &
#
#rm test_cartpole_a3c/a3c-18 -rf
#python -B oo_a3c.py CartPole-v1 --model ActorCriticModel --agent ActorCriticAgent --learning-rate 0.001 --target-network-update 20 --egreedy-final 0.01 --egreedy-decay 0.9999 0.999 0.998 --egreedy-props 0.4 0.3 0.3 --replay-start-size 1000 --replay-buffer-size 20000 --batch-size 64 --update-frequency 1 --thread-count 8 --nstep 1000 --max-step 600000 --shared-model True --gpu 1 --logdir test_cartpole_a3c/a3c-18 &
#
#rm test_cartpole_a3c/a3c-17 -rf
#python -B oo_a3c.py CartPole-v1 --model ActorCriticModel --agent ActorCriticAgent --learning-rate 0.001 --target-network-update 20 --egreedy-final 0.01 --egreedy-decay 0.9999 0.999 0.998 --egreedy-props 0.4 0.3 0.3 --replay-start-size 1000 --replay-buffer-size 20000 --batch-size 64 --update-frequency 1 --thread-count 8 --nstep 1000 --max-step 600000 --shared-model True --gpu 1 --logdir test_cartpole_a3c/a3c-17 &
#
#rm test_cartpole_a3c/a3c-16 -rf
#python -B oo_a3c.py CartPole-v1 --model ActorCriticModel --agent ActorCriticAgent --learning-rate 0.001 --target-network-update 20 --egreedy-final 0.01 --egreedy-decay 0.9999 0.999 0.998 --egreedy-props 0.4 0.3 0.3 --replay-start-size 1000 --replay-buffer-size 20000 --batch-size 64 --update-frequency 1 --thread-count 8 --nstep 1000 --max-step 600000 --shared-model True --gpu 1 --logdir test_cartpole_a3c/a3c-16 &

#rm test_cartpole_a3c/a3c-15 -rf
#python -B oo_a3c.py CartPole-v1 --model ActorCriticModel --agent ActorCriticAgent --learning-rate 0.001 --target-network-update 20 --egreedy-final 0.01 --egreedy-decay 0.9999 0.999 0.998 --egreedy-props 0.4 0.3 0.3 --replay-start-size 1000 --replay-buffer-size 20000 --batch-size 64 --update-frequency 1 --thread-count 8 --nstep 1000 --max-step 600000 --shared-model True --logdir test_cartpole_a3c/a3c-15

# cpu third test (reward not dropped for the second try)
#rm test_cartpole_a3c/a3c-14 -rf
#python -B oo_a3c.py CartPole-v1 --model ActorCriticModel --agent ActorCriticAgent --learning-rate 0.001 --target-network-update 20 --egreedy-final 0.01 --egreedy-decay 0.9999 0.999 0.998 --egreedy-props 0.4 0.3 0.3 --replay-start-size 1000 --replay-buffer-size 20000 --batch-size 64 --update-frequency 1 --thread-count 8 --nstep 1000 --max-step 600000 --shared-model True --logdir test_cartpole_a3c/a3c-14

# cpu second test (reward dropped for the first try)
#rm test_cartpole_a3c/a3c-13 -rf
#python -B oo_a3c.py CartPole-v1 --model ActorCriticModel --agent ActorCriticAgent --learning-rate 0.001 --target-network-update 20 --egreedy-final 0.01 --egreedy-decay 0.9999 0.999 0.998 --egreedy-props 0.4 0.3 0.3 --replay-start-size 1000 --replay-buffer-size 20000 --batch-size 64 --update-frequency 1 --thread-count 8 --nstep 1000 --max-step 600000 --shared-model True --logdir test_cartpole_a3c/a3c-13

# use cpu again (same with a3c-10)
#rm test_cartpole_a3c/a3c-12 -rf
#python -B oo_a3c.py CartPole-v1 --model ActorCriticModel --agent ActorCriticAgent --learning-rate 0.001 --target-network-update 20 --egreedy-final 0.01 --egreedy-decay 0.9999 0.999 0.998 --egreedy-props 0.4 0.3 0.3 --replay-start-size 1000 --replay-buffer-size 20000 --batch-size 64 --update-frequency 1 --thread-count 8 --nstep 1000 --max-step 600000 --shared-model True --logdir test_cartpole_a3c/a3c-12

# use gpu
#rm test_cartpole_a3c/a3c-11 -rf
#python -B oo_a3c.py CartPole-v1 --model ActorCriticModel --agent ActorCriticAgent --learning-rate 0.001 --target-network-update 20 --egreedy-final 0.01 --egreedy-decay 0.9999 0.999 0.998 --egreedy-props 0.4 0.3 0.3 --replay-start-size 1000 --replay-buffer-size 20000 --batch-size 64 --update-frequency 1 --thread-count 8 --nstep 1000 --max-step 600000 --shared-model True --gpu 1 --logdir test_cartpole_a3c/a3c-11

# shared_model added
#rm test_cartpole_a3c/a3c-10 -rf
#python -B oo_a3c.py CartPole-v1 --model ActorCriticModel --agent ActorCriticAgent --learning-rate 0.001 --target-network-update 20 --egreedy-final 0.01 --egreedy-decay 0.9999 0.999 0.998 --egreedy-props 0.4 0.3 0.3 --replay-start-size 1000 --replay-buffer-size 20000 --batch-size 64 --update-frequency 1 --thread-count 8 --nstep 1000 --max-step 440000 --shared-model True --logdir test_cartpole_a3c/a3c-10

# misc tests
#rm test_cartpole_a3c/a3c-9 -rf
#python -B oo_a3c.py CartPole-v1 --model ActorCriticModel --agent ActorCriticAgent --learning-rate 0.001 --target-network-update 20 --egreedy-final 0.01 --egreedy-decay 0.9999 0.999 0.998 --egreedy-props 0.4 0.3 0.3 --replay-start-size 1000 --replay-buffer-size 20000 --batch-size 64 --update-frequency 1 --thread-count 1 --nstep 1000 --max-step 440000 --logdir test_cartpole_a3c/a3c-9

# stop_gradient is added to advantage placeholder (don't now why it is needed if it is), maxstep increased
#rm test_cartpole_a3c/a3c-8 -rf
#python -B oo_a3c.py CartPole-v1 --model ActorCriticModel --agent ActorCriticAgent --learning-rate 0.001 --target-network-update 20 --egreedy-final 0.01 --egreedy-decay 0.9999 0.999 0.998 --egreedy-props 0.4 0.3 0.3 --replay-start-size 1000 --replay-buffer-size 20000 --batch-size 64 --update-frequency 1 --thread-count 8 --nstep 1000 --max-step 440000 --logdir test_cartpole_a3c/a3c-8

# network initialization changed
#rm test_cartpole_a3c/a3c-7 -rf
#python -B oo_a3c.py CartPole-v1 --model ActorCriticModel --agent ActorCriticAgent --learning-rate 0.001 --target-network-update 20 --egreedy-final 0.01 --egreedy-decay 0.9999 0.999 0.998 --egreedy-props 0.4 0.3 0.3 --replay-start-size 1000 --replay-buffer-size 20000 --batch-size 64 --update-frequency 1 --thread-count 8 --nstep 1000 --max-step 320000 --logdir test_cartpole_a3c/a3c-7

# network changed. only first layer shared, actor and critic has their own hidden layer
#rm test_cartpole_a3c/a3c-6 -rf
#python -B oo_a3c.py CartPole-v1 --model ActorCriticModel --agent ActorCriticAgent --learning-rate 0.001 --target-network-update 20 --egreedy-final 0.01 --egreedy-decay 0.9999 0.999 0.998 --egreedy-props 0.4 0.3 0.3 --replay-start-size 1000 --replay-buffer-size 20000 --batch-size 64 --update-frequency 1 --thread-count 8 --nstep 1000 --max-step 320000 --logdir test_cartpole_a3c/a3c-6

# comparison with rlcode: when step_count<500 calculate next value (in agent.py, marked by @ersin)
#rm test_cartpole_a3c/a3c-5 -rf
#python -B oo_a3c.py CartPole-v1 --model ActorCriticModel --agent ActorCriticAgent --learning-rate 0.001 --target-network-update 20 --egreedy-final 0.01 --egreedy-decay 0.9999 0.999 0.998 --egreedy-props 0.4 0.3 0.3 --replay-start-size 1000 --replay-buffer-size 20000 --batch-size 64 --update-frequency 1 --thread-count 8 --nstep 1000 --max-step 320000 --logdir test_cartpole_a3c/a3c-5

# comparison with rlcode: only train at the end of episode
#rm test_cartpole_a3c/a3c-4 -rf
#python -B oo_a3c.py CartPole-v1 --model ActorCriticModel --agent ActorCriticAgent --learning-rate 0.001 --target-network-update 20 --egreedy-final 0.01 --egreedy-decay 0.9999 0.999 0.998 --egreedy-props 0.4 0.3 0.3 --replay-start-size 1000 --replay-buffer-size 20000 --batch-size 64 --update-frequency 1 --thread-count 8 --nstep 1000 --max-step 320000 --logdir test_cartpole_a3c/a3c-4

# disabling Penalizer
#rm test_cartpole_a3c/a3c-3 -rf
#python -B oo_a3c.py CartPole-v1 --model ActorCriticModel --agent ActorCriticAgent --learning-rate 0.001 --target-network-update 20 --egreedy-final 0.01 --egreedy-decay 0.9999 0.999 0.998 --egreedy-props 0.4 0.3 0.3 --replay-start-size 1000 --replay-buffer-size 20000 --batch-size 64 --update-frequency 1 --thread-count 8 --max-step 320000 --logdir test_cartpole_a3c/a3c-3

# using shared network
#rm test_cartpole_a3c/a3c-2 -rf
#python -B oo_a3c.py CartPole-v1 --model ActorCriticModel --agent ActorCriticAgent --learning-rate 0.001 --target-network-update 20 --egreedy-final 0.01 --egreedy-decay 0.9999 0.999 0.998 --egreedy-props 0.4 0.3 0.3  --env-transforms Penalizer --replay-start-size 1000 --replay-buffer-size 20000 --batch-size 64 --update-frequency 1 --thread-count 8 --max-step 320000 --logdir test_cartpole_a3c/a3c-2

# using different network per thread
#rm test_cartpole_a3c/a3c-1 -rf
#python -B oo_a3c.py CartPole-v1 --model ActorCriticModel --agent ActorCriticAgent --learning-rate 0.001 --target-network-update 20 --egreedy-final 0.01 --egreedy-decay 0.9999 0.999 0.998 --egreedy-props 0.4 0.3 0.3  --env-transforms Penalizer --replay-start-size 1000 --replay-buffer-size 20000 --batch-size 64 --update-frequency 1 --thread-count 8 --max-step 320000 --logdir test_cartpole_a3c/a3c-1

#python -B oo_dqn.py CartPole-v1 --model CartPoleModel --learning-rate 0.001 --target-network-update 20 --egreedy-final 0.01 --egreedy-decay 0.999 --env-transforms Penalizer --replay-start-size 1000 --replay-buffer-size 20000 --batch-size 64 --update-frequency 1 --max-step 50000 --logdir test_cartpole2/dqn-4

#python -B oo_a3c.py CartPole-v1 --model CartPoleModel --nstep 5 --learning-rate 0.01 --target-network-update 20 --egreedy-final 0.01 --egreedy-decay 0.9999 0.999 0.998 --egreedy-props 0.4 0.3 0.3 --env-transforms Penalizer --replay-start-size 64 --replay-buffer-size 2000 --batch-size 64 --update-frequency 1 --thread-count 16 --max-step 200000 --logdir test_cartpole2/a3c-10

# the following lines are tests

#python -B oo_a3c.py CartPole-v1 --model CartPoleModel --nstep 5 --learning-rate 0.001 --target-network-update 20 --egreedy-final 0.001 --egreedy-props 1 --egreedy-final-step 6000 --egreedy-decay 1 --env-transforms Penalizer --replay-start-size 64 --replay-buffer-size 2000 --batch-size 64 --update-frequency 1 --thread-count 16 --logdir test_cartpole/a3c-t16-n5-ed1-efs6000-8

#python -B oo_a3c.py CartPole-v1 --model CartPoleModel --nstep 0 --learning-rate 0.001 --target-network-update 20 --egreedy-final 0.01 --egreedy-decay 0.995 --env-transforms Penalizer --replay-start-size 64 --replay-buffer-size 2000 --batch-size 64 --update-frequency 1 --thread-count 16 --logdir test_cartpole/a3c-t16-n0-ed0.998-7
