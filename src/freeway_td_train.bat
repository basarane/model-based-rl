REM 256x256x256x1
rm freeway_td/train13 -rf
pipenv run python -B oo_td.py FreewayDeterministic-v4 --batch-size 4 --max-step 20000 --learning-rate 0.001 --target-network-update 200 --sample-count 50 --env-model EnvModelFreewayManual --vmodel FreewayVNetwork --env-transforms FreewayTransform --save-freq 100 --logdir freeway_td/train13

REM RMSprop
REM rm freeway_td/train12 -rf
REM pipenv run python -B oo_td.py FreewayDeterministic-v4 --batch-size 4 --max-step 20000 --learning-rate 0.001 --target-network-update 200 --sample-count 50 --env-model EnvModelFreewayManual --vmodel FreewayVNetwork --env-transforms FreewayTransform --save-freq 100 --logdir freeway_td/train12

REM amsgrad=True kalmis 10'da. False yapip tekrar calistirdim
REM rm freeway_td/train11 -rf
REM pipenv run python -B oo_td.py FreewayDeterministic-v4 --batch-size 4 --max-step 20000 --learning-rate 0.001 --target-network-update 200 --sample-count 50 --env-model EnvModelFreewayManual --vmodel FreewayVNetwork --env-transforms FreewayTransform --save-freq 100 --logdir freeway_td/train11

REM adam 7'nin tekrari
REM rm freeway_td/train10 -rf
REM pipenv run python -B oo_td.py FreewayDeterministic-v4 --batch-size 4 --max-step 20000 --learning-rate 0.001 --target-network-update 200 --sample-count 50 --env-model EnvModelFreewayManual --vmodel FreewayVNetwork --env-transforms FreewayTransform --save-freq 100 --logdir freeway_td/train10

REM Adam amsgrad
REM rm freeway_td/train9 -rf
REM pipenv run python -B oo_td.py FreewayDeterministic-v4 --batch-size 4 --max-step 20000 --learning-rate 0.001 --target-network-update 200 --sample-count 50 --env-model EnvModelFreewayManual --vmodel FreewayVNetwork --env-transforms FreewayTransform --save-freq 100 --logdir freeway_td/train9

REM Nadam
REM rm freeway_td/train8 -rf
REM pipenv run python -B oo_td.py FreewayDeterministic-v4 --batch-size 4 --max-step 20000 --learning-rate 0.001 --target-network-update 200 --sample-count 50 --env-model EnvModelFreewayManual --vmodel FreewayVNetwork --env-transforms FreewayTransform --save-freq 100 --logdir freeway_td/train8

REM 200 sample, 128x128x1
REM rm freeway_td/train7 -rf
REM pipenv run python -B oo_td.py FreewayDeterministic-v4 --batch-size 4 --max-step 20000 --learning-rate 0.001 --target-network-update 200 --sample-count 50 --env-model EnvModelFreewayManual --vmodel FreewayVNetwork --env-transforms FreewayTransform --save-freq 100 --logdir freeway_td/train7

REM cross: 0.2 carpisma: 0.3, 512x512x1
REM rm freeway_td/train6 -rf
REM pipenv run python -B oo_td.py FreewayDeterministic-v4 --batch-size 4 --max-step 20000 --learning-rate 0.001 --target-network-update 200 --sample-count 20 --env-model EnvModelFreewayManual --vmodel FreewayVNetwork --env-transforms FreewayTransform --save-freq 100 --logdir freeway_td/train6

REM TDNetwork'de debug acildi
REM rm freeway_td/train5 -rf
REM pipenv run python -B oo_td.py FreewayDeterministic-v4 --batch-size 4 --max-step 10 --learning-rate 0.001 --target-network-update 200 --sample-count 20 --env-model EnvModelFreewayManual --vmodel FreewayVNetwork --env-transforms FreewayTransform --save-freq 100 --logdir freeway_td/train5


REM 512x128x1
REM rm freeway_td/train4 -rf
REM pipenv run python -B oo_td.py FreewayDeterministic-v4 --batch-size 4 --max-step 20000 --learning-rate 0.001 --target-network-update 200 --sample-count 20 --env-model EnvModelFreewayManual --vmodel FreewayVNetwork --env-transforms FreewayTransform --save-freq 100 --logdir freeway_td/train4

REM 2 layer mlp 512x1 (onceki 512x512x512x1 idi
REM rm freeway_td/train3 -rf
REM pipenv run python -B oo_td.py FreewayDeterministic-v4 --batch-size 4 --max-step 20000 --learning-rate 0.001 --target-network-update 200 --sample-count 20 --env-model EnvModelFreewayManual --vmodel FreewayVNetwork --env-transforms FreewayTransform --save-freq 100 --logdir freeway_td/train3
