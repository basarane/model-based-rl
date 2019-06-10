rmdir freeway_td\test26c /s /q
pipenv run python -B oo_td_test.py FreewayDeterministic-v4 --load-weightfile freeway_td/train26/weights_ 500 500 13000 --env-model EnvModelFreewayManual --vmodel FreewayVNetwork --env-transforms FreewayTransform --max-episode 4 --logdir freeway_td/test26c

REM rmdir freeway_td\test26 /s /q
REM pipenv run python -B oo_td_test.py FreewayDeterministic-v4 --load-weightfile freeway_td/train26/weights_ 1000 1000 13000 --env-model EnvModelFreewayManual --vmodel FreewayVNetwork --env-transforms FreewayTransform --max-episode 1 --logdir freeway_td/test26

REM rm freeway_td/test25 -rf
REM pipenv run python -B oo_td_test.py FreewayDeterministic-v4 --load-weightfile freeway_td/train25/weights_ 9000 100 10000 --env-model EnvModelFreewayManual --vmodel FreewayVNetwork --env-transforms FreewayTransform --enable-render True --render-step 4 --logdir freeway_td/test25

REM rm freeway_td/test24 -rf
REM pipenv run python -B oo_td_test.py FreewayDeterministic-v4 --load-weightfile freeway_td/train24/weights_ 5500 100 6000 --env-model EnvModelFreewayManual --vmodel FreewayVNetwork --env-transforms FreewayTransform --enable-render True --render-step 4 --logdir freeway_td/test24

REM rm freeway_td/test22 -rf
REM pipenv run python -B oo_td_test.py FreewayDeterministic-v4 --load-weightfile freeway_td/train22/weights_ 23000 100 24000 --env-model EnvModelFreewayManual --vmodel FreewayVNetwork --env-transforms FreewayTransform --enable-render True --render-step 4 --logdir freeway_td/test22

REM rm freeway_td/test21 -rf
REM pipenv run python -B oo_td_test.py FreewayDeterministic-v4 --load-weightfile freeway_td/train21/weights_ 27700 100 28000 --env-model EnvModelFreewayManual --vmodel FreewayVNetwork --env-transforms FreewayTransform --enable-render True --render-step 4 --logdir freeway_td/test21

REM rm freeway_td/test20 -rf
REM pipenv run python -B oo_td_test.py FreewayDeterministic-v4 --load-weightfile freeway_td/train20/weights_ 15000 100 15500 --env-model EnvModelFreewayManual --vmodel FreewayVNetwork --env-transforms FreewayTransform --enable-render True --render-step 4 --logdir freeway_td/test20

REM rm freeway_td/test19 -rf
REM pipenv run python -B oo_td_test.py FreewayDeterministic-v4 --load-weightfile freeway_td/train19/weights_ 7300 100 7500 --env-model EnvModelFreewayManual --vmodel FreewayVNetwork --env-transforms FreewayTransform --enable-render True --render-step 4 --logdir freeway_td/test19

REM rm freeway_td/test18 -rf
REM pipenv run python -B oo_td_test.py FreewayDeterministic-v4 --load-weightfile freeway_td/train18/weights_ 20000 100 53000 --env-model EnvModelFreewayManual --vmodel FreewayVNetwork --env-transforms FreewayTransform --enable-render True --render-step 4 --logdir freeway_td/test18


REM rm freeway_td/test13 -rf
REM pipenv run python -B oo_td_test.py FreewayDeterministic-v4 --load-weightfile freeway_td/train13/weights_ 14000 100 15000 --env-model EnvModelFreewayManual --vmodel FreewayVNetwork --env-transforms FreewayTransform --enable-render True --render-step 4 --logdir freeway_td/test13

REM 2 layer mlp 512x1 (onceki 512x512x512x1 idi
REM rm freeway_td/test3 -rf
REM pipenv run python -B oo_td_test.py FreewayDeterministic-v4 --load-weightfile freeway_td/train3/weights_ 3000 100 3300 --env-model EnvModelFreewayManual --vmodel FreewayVNetwork --env-transforms FreewayTransform --enable-render True --render-step 4 --logdir freeway_td/test3
