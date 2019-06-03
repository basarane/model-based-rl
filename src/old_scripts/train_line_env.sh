#sadece env model degisti, artik [0.45, 0.55] arasinda episode bitiyor
rm test_line_env/lineenv-15 -rf
python -B oo_env.py Line --mode train --env-model EnvModelLine --model LineModel --learning-rate 0.005 --batch-size 64 --max-step 20000 --save-interval 100 --load-trajectory test_line_dqn/traj-3b_mix.h5 --logdir test_line_env/lineenv-15

# tekrar relu
#rm test_line_env/lineenv-14 -rf
#python -B oo_env.py Line --mode train --env-model EnvModelLine --model LineModel --learning-rate 0.005 --batch-size 64 --max-step 20000 --save-interval 100 --load-trajectory test_line_dqn/traj-2b_mix.h5 --logdir test_line_env/lineenv-14

# single layer sigmoid'e donuldu, full random 2000 step'den full random 20000 step'e cikildi, batch size telrar 64 yapildi
#rm test_line_env/lineenv-13 -rf
#python -B oo_env.py Line --mode train --env-model EnvModelLine --model LineModel --learning-rate 0.005 --batch-size 64 --max-step 20000 --save-interval 100 --load-trajectory test_line_dqn/traj-2b_mix.h5 --logdir test_line_env/lineenv-13

# rbf yerine 20 RELU + 20 hard_sigmoid yapildi
#rm test_line_env/lineenv-11 -rf
#python -B oo_env.py Line --mode train --env-model EnvModelLine --model LineModel --learning-rate 0.005 --batch-size 32 --max-step 20000 --save-interval 100 --load-trajectory test_line_dqn/traj-2b_mix.h5 --logdir test_line_env/lineenv-11

# rbf sayisi 10'dan 100'e cikartildi, beta 0.3'den 0.02'e dusuruldu
#rm test_line_env/lineenv-10 -rf
#python -B oo_env.py Line --mode train --env-model EnvModelLine --model LineModel --learning-rate 0.005 --batch-size 32 --max-step 20000 --save-interval 100 --load-trajectory test_line_dqn/traj-2b_mix.h5 --logdir test_line_env/lineenv-10

# adam'a donuldu
#rm test_line_env/lineenv-9 -rf
#python -B oo_env.py Line --mode train --env-model EnvModelLine --model LineModel --learning-rate 0.005 --batch-size 32 --max-step 20000 --save-interval 100 --load-trajectory test_line_dqn/traj-2b_mix.h5 --logdir test_line_env/lineenv-9

#RBF layer'i eklendi, batch size arttirildi
#rm test_line_env/lineenv-8 -rf
#python -B oo_env.py Line --mode train --env-model EnvModelLine --model LineModel --learning-rate 0.005 --batch-size 32 --max-step 20000 --save-interval 100 --load-trajectory test_line_dqn/traj-2b_mix.h5 --logdir test_line_env/lineenv-8

# sigmoide gecildi, batchsize azaldilti, sgd'e gecildi
#rm test_line_env/lineenv-7 -rf
#python -B oo_env.py Line --mode train --env-model EnvModelLine --model LineModel --learning-rate 0.005 --batch-size 8 --max-step 20000 --save-interval 100 --load-trajectory test_line_dqn/traj-2b_mix.h5 --logdir test_line_env/lineenv-7

# full random trajectory ile egitildi
#rm test_line_env/lineenv-6 -rf
#python -B oo_env.py Line --mode train --env-model EnvModelLine --model LineModel --learning-rate 0.005 --batch-size 64 --max-step 20000 --save-interval 100 --load-trajectory test_line_dqn/traj-2b_mix.h5 --logdir test_line_env/lineenv-6

# tanh tekrar relu yapildi, ADAM'a geri donuldu
#rm test_line_env/lineenv-5 -rf
#python -B oo_env.py Line --mode train --env-model EnvModelLine --model LineModel --learning-rate 0.005 --batch-size 64 --max-step 20000 --save-interval 100 --load-trajectory test_line_dqn/traj-2_mix.h5 --logdir test_line_env/lineenv-5

# reward modulundeki hidden layer her bir action icin de ayrildi ve direk inputa baglandi (tanh layer'i), RMSprop'a gecildi
#rm test_line_env/lineenv-4 -rf
#python -B oo_env.py Line --mode train --env-model EnvModelLine --model LineModel --learning-rate 0.005 --batch-size 64 --max-step 20000 --save-interval 100 --load-trajectory test_line_dqn/traj-2_mix.h5 --logdir test_line_env/lineenv-4

# sadece reward modulune bir ara tanh layer'i eklendi, reward weight'i 5'e cikartildi
#rm test_line_env/lineenv-3 -rf
#python -B oo_env.py Line --mode train --env-model EnvModelLine --model LineModel --learning-rate 0.005 --batch-size 64 --max-step 20000 --save-interval 100 --load-trajectory test_line_dqn/traj-2_mix.h5 --logdir test_line_env/lineenv-3

#rm test_line_env/lineenv-2 -rf
#python -B oo_env.py Line --mode train --env-model EnvModelLine --model LineModel --learning-rate 0.005 --batch-size 64 --max-step 20000 --save-interval 100 --load-trajectory test_line_dqn/traj-2_mix.h5 --logdir test_line_env/lineenv-2

# sadece train suresi uzatildi: 300k
#rm test_line_env/lineenv-1 -rf
#python -B oo_env.py Line --mode train --env-model EnvModelLine --model LineModel --learning-rate 0.001 --batch-size 64 --max-step 5000 --save-interval 100 --load-trajectory test_line_dqn/traj-1_mix.h5 --logdir test_line_env/lineenv-1
