
python -B td_v_change_line.py --basedir td_offline_v_change_line_1 --prefix algo_convergence_line/td/train-0/weights_ --idx 100 100 50001 &
python -B td_v_change_line.py --basedir td_offline_v_change_line_2 --prefix algo_convergence_line/td/train-1/weights_ --idx 100 100 50001 &
python -B td_v_change_line.py --basedir td_offline_v_change_line_3 --prefix algo_convergence_line/td/train-2/weights_ --idx 100 100 50001 &
python -B td_v_change_line.py --basedir td_offline_v_change_line_4 --prefix algo_convergence_line/td/train-3/weights_ --idx 100 100 50001 &

python -B td_v_change_line.py --basedir td_online_v_change_line_1 --prefix algo_convergence_line/td_realtime/train-0/weights_ --idx 100 100 50001 &
python -B td_v_change_line.py --basedir td_online_v_change_line_2 --prefix algo_convergence_line/td_realtime/train-1/weights_ --idx 100 100 50001 &
python -B td_v_change_line.py --basedir td_online_v_change_line_3 --prefix algo_convergence_line/td_realtime/train-2/weights_ --idx 100 100 50001 &
python -B td_v_change_line.py --basedir td_online_v_change_line_4 --prefix algo_convergence_line/td_realtime/train-3/weights_ --idx 100 100 50001 &

#python -B td_v_change_line.py --basedir td_offline_v_change_line_23 --prefix ../test_line_td/linetd-23/weights_ --idx 10 10 50001 &
#python -B td_v_change_line.py --basedir td_offline_v_change_line_22 --prefix ../test_line_td/linetd-22/weights_ --idx 10 10 50001 &
#python -B td_v_change_line.py --basedir td_offline_v_change_line_21 --prefix ../test_line_td/linetd-21/weights_ --idx 10 10 50001 &
#python -B td_v_change_line.py --basedir td_offline_v_change_line_20 --prefix ../test_line_td/linetd-20/weights_ --idx 10 10 50001 &
#python -B td_v_change_line.py --basedir td_offline_v_change_line_19 --prefix ../test_line_td/linetd-19/weights_ --idx 10 10 50001 &
#
#python -B td_v_change_line.py --basedir td_offline_v_change_line_18 --prefix ../test_line_td/linetd-18/weights_ --idx 1000 1000 50000 &
#python -B td_v_change_line.py --basedir td_offline_v_change_line_17 --prefix ../test_line_td/linetd-17/weights_ --idx 1000 1000 50000 &
#python -B td_v_change_line.py --basedir td_offline_v_change_line_16 --prefix ../test_line_td/linetd-16/weights_ --idx 1000 1000 50000 &
#python -B td_v_change_line.py --basedir td_offline_v_change_line_15 --prefix ../test_line_td/linetd-15/weights_ --idx 1000 1000 50000 &
#python -B td_v_change_line.py --basedir td_offline_v_change_line_14 --prefix ../test_line_td/linetd-14/weights_ --idx 1000 1000 50000 &
#
