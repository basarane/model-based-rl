
# hidden 1024x1024, activation sigmoid, optimizer Adam
rm test_lander_model/reward-5 -rf
python -B oo_env.py LunarLander-v2 --mode train --model LunarLanderModel --env-model EnvModelLunarLander2 --learning-rate 0.00025 --batch-size 64 --max-step 100000 --save-interval 5000 --load-trajectory test_lander_model/traj-1_mix.h5 --logdir test_lander_model/reward-5

# instead of 256x256 hidden, single 1024
#rm test_lander_model/reward-4 -rf
#python -B oo_env.py LunarLander-v2 --mode train --model LunarLanderModel --env-model EnvModelLunarLander2 --learning-rate 0.0001 --batch-size 64 --max-step 100000 --save-interval 5000 --load-trajectory test_lander_model/traj-1_mix.h5 --logdir test_lander_model/reward-4


# new model: EnvModelLunarLander2, no shared hidden layer
#rm test_lander_model/reward-3 -rf
#python -B oo_env.py LunarLander-v2 --mode train --model LunarLanderModel --env-model EnvModelLunarLander2 --learning-rate 0.0001 --batch-size 64 --max-step 100000 --save-interval 5000 --load-trajectory test_lander_model/traj-1_mix.h5 --logdir test_lander_model/reward-3

# model changed: 256 unit hidden layers added after shared hidden layer, initializer is changed from he_uniform to default
#rm test_lander_model/reward-2 -rf
#python -B oo_env.py LunarLander-v2 --mode train --model LunarLanderModel --env-model EnvModelLunarLander --learning-rate 0.001 --batch-size 64 --max-step 100000 --save-interval 5000 --load-trajectory test_lander_model/traj-1_mix.h5 --logdir test_lander_model/reward-2

# first lunar lander test
#rm test_lander_model/reward-1 -rf
#python -B oo_env.py LunarLander-v2 --mode train --model LunarLanderModel --env-model EnvModelLunarLander --learning-rate 0.001 --batch-size 64 --max-step 100000 --save-interval 5000 --load-trajectory test_lander_model/traj-1_mix.h5 --logdir test_lander_model/reward-1

