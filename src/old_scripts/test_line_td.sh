rm test_line_v/line_v-24 -rf
python -B oo_td_test.py Line --env-model EnvModelLineManual --max-episode 5 --vmodel LineVNetwork --load-weightfile  test_line_td/linetd-24/weights_ 1000 1000 50001 --logdir test_line_v/line_v-24

# 5 tests after sampling using square of td error
#rm test_line_v/line_v-23 -rf
#python -B oo_td_test.py Line --env-model EnvModelLineManual --max-episode 5 --vmodel LineVNetwork --load-weightfile  test_line_td/linetd-23/weights_ 100 100 50001 --logdir test_line_v/line_v-23 &
#
#rm test_line_v/line_v-22 -rf
#python -B oo_td_test.py Line --env-model EnvModelLineManual --max-episode 5 --vmodel LineVNetwork --load-weightfile  test_line_td/linetd-22/weights_ 100 100 50001 --logdir test_line_v/line_v-22 &
#
#rm test_line_v/line_v-21 -rf
#python -B oo_td_test.py Line --env-model EnvModelLineManual --max-episode 5 --vmodel LineVNetwork --load-weightfile  test_line_td/linetd-21/weights_ 100 100 50001 --logdir test_line_v/line_v-21 &
#
#rm test_line_v/line_v-20 -rf
#python -B oo_td_test.py Line --env-model EnvModelLineManual --max-episode 5 --vmodel LineVNetwork --load-weightfile  test_line_td/linetd-20/weights_ 100 100 50001 --logdir test_line_v/line_v-20 &
#
#rm test_line_v/line_v-19 -rf
#python -B oo_td_test.py Line --env-model EnvModelLineManual --max-episode 5 --vmodel LineVNetwork --load-weightfile  test_line_td/linetd-19/weights_ 100 100 50001 --logdir test_line_v/line_v-19 &

# 5 tests after sampling using absolute td error
#rm test_line_v/line_v-18 -rf
#python -B oo_td_test.py Line --env-model EnvModelLineManual --max-episode 5 --vmodel LineVNetwork --load-weightfile  test_line_td/linetd-18/weights_ 100 100 50001 --logdir test_line_v/line_v-18 &
#
#rm test_line_v/line_v-17 -rf
#python -B oo_td_test.py Line --env-model EnvModelLineManual --max-episode 5 --vmodel LineVNetwork --load-weightfile  test_line_td/linetd-17/weights_ 100 100 50001 --logdir test_line_v/line_v-17 &
#
#rm test_line_v/line_v-16 -rf
#python -B oo_td_test.py Line --env-model EnvModelLineManual --max-episode 5 --vmodel LineVNetwork --load-weightfile  test_line_td/linetd-16/weights_ 100 100 50001 --logdir test_line_v/line_v-16 &
#
#rm test_line_v/line_v-15 -rf
#python -B oo_td_test.py Line --env-model EnvModelLineManual --max-episode 5 --vmodel LineVNetwork --load-weightfile  test_line_td/linetd-15/weights_ 100 100 50001 --logdir test_line_v/line_v-15 &
#
#rm test_line_v/line_v-14 -rf
#python -B oo_td_test.py Line --env-model EnvModelLineManual --max-episode 5 --vmodel LineVNetwork --load-weightfile  test_line_td/linetd-14/weights_ 100 100 50001 --logdir test_line_v/line_v-14 &

# test 5 td runs in parallel
#rm test_line_v/line_v-9 -rf
#python -B oo_td_test.py Line --env-model EnvModelLineManual --max-episode 5 --vmodel LineVNetwork --load-weightfile  test_line_td/linetd-9/weights_ 100 100 10000 --logdir test_line_v/line_v-9 &
#
#rm test_line_v/line_v-8 -rf
#python -B oo_td_test.py Line --env-model EnvModelLineManual --max-episode 5 --vmodel LineVNetwork --load-weightfile  test_line_td/linetd-8/weights_ 100 100 10000 --logdir test_line_v/line_v-8 &
#
#rm test_line_v/line_v-7 -rf
#python -B oo_td_test.py Line --env-model EnvModelLineManual --max-episode 5 --vmodel LineVNetwork --load-weightfile  test_line_td/linetd-7/weights_ 100 100 10000 --logdir test_line_v/line_v-7 &
#
#rm test_line_v/line_v-6 -rf
#python -B oo_td_test.py Line --env-model EnvModelLineManual --max-episode 5 --vmodel LineVNetwork --load-weightfile  test_line_td/linetd-6/weights_ 100 100 10000 --logdir test_line_v/line_v-6 &
#
#rm test_line_v/line_v-5 -rf
#python -B oo_td_test.py Line --env-model EnvModelLineManual --max-episode 5 --vmodel LineVNetwork --load-weightfile  test_line_td/linetd-5/weights_ 100 100 10000 --logdir test_line_v/line_v-5 &

#rm test_line_v/line_v-2 -rf
#python -B oo_td_test.py Line --env-model EnvModelLine --env-weightfile test_line_env/lineenv-2/weights_20000.h5 --max-episode 5 --vmodel LineVNetwork --load-weightfile  test_line_td/linetd-2/weights_ 1 1 1000 --logdir test_line_v/line_v-2

# bir onceki ile ayni. VAgent'ta sadece next state'e bakiliyordu, artik reward + next_state'e bakiliyor 
#mkdir test_line_v
#rm test_line_v/line_v-1 -rf
#python -B oo_td_test.py Line --env-model EnvModelLine --env-weightfile test_line_env/lineenv-1/weights_5000.h5 --max-episode 5 --vmodel LineVNetwork --load-weightfile  test_line_td/linetd-1/weights_ 1 1 1000 --logdir test_line_v/line_v-1
