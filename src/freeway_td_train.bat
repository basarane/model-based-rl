REM cross: 0.1, carpisma: 0.2
rm freeway_td/train26 -rf
pipenv run python -B oo_td.py FreewayDeterministic-v4 --batch-size 4 --max-step 100000 --learning-rate 0.001 --target-network-update 200 --sample-count 10 --env-model EnvModelFreewayManual --vmodel FreewayVNetwork --env-transforms FreewayTransform --save-freq 100 --logdir freeway_td/train26

REM hidden 256'ya cikartildi, carpisma ve cross timer'larin degerleri reverse edildi, sifirlanan her sey kaldirildi (sanirim)
REM rm freeway_td/train25 -rf
REM pipenv run python -B oo_td.py FreewayDeterministic-v4 --batch-size 4 --max-step 100000 --learning-rate 0.001 --target-network-update 200 --sample-count 10 --env-model EnvModelFreewayManual --vmodel FreewayVNetwork --env-transforms FreewayTransform --save-freq 100 --logdir freeway_td/train25

REM 32'lik hidden layer eklendi
REM rm freeway_td/train24 -rf
REM pipenv run python -B oo_td.py FreewayDeterministic-v4 --batch-size 4 --max-step 100000 --learning-rate 0.001 --target-network-update 200 --sample-count 10 --env-model EnvModelFreewayManual --vmodel FreewayVNetwork --env-transforms FreewayTransform --save-freq 100 --logdir freeway_td/train24

REM pozisyonlar 0 ile 1 arasinda sayilara cevrilip tek input yapildi
REM rm freeway_td/train23 -rf
REM pipenv run python -B oo_td.py FreewayDeterministic-v4 --batch-size 4 --max-step 100000 --learning-rate 0.001 --target-network-update 200 --sample-count 10 --env-model EnvModelFreewayManual --vmodel FreewayVNetwork --env-transforms FreewayTransform --save-freq 100 --logdir freeway_td/train23

REM tavuk channel'lari K.sum ile toplanip VModel'e verildi
REM rm freeway_td/train22 -rf
REM pipenv run python -B oo_td.py FreewayDeterministic-v4 --batch-size 4 --max-step 100000 --learning-rate 0.001 --target-network-update 200 --sample-count 10 --env-model EnvModelFreewayManual --vmodel FreewayVNetwork --env-transforms FreewayTransform --save-freq 100 --logdir freeway_td/train22

REM network x1'e dusuruldu, tum hidden layer'lar atildi
REM rm freeway_td/train21 -rf
REM pipenv run python -B oo_td.py FreewayDeterministic-v4 --batch-size 4 --max-step 100000 --learning-rate 0.001 --target-network-update 200 --sample-count 10 --env-model EnvModelFreewayManual --vmodel FreewayVNetwork --env-transforms FreewayTransform --save-freq 100 --logdir freeway_td/train21

REM sadece tavuk birakildi, arabalar ve timer'lar sifirla carpildi
REM rm freeway_td/train20 -rf
REM pipenv run python -B oo_td.py FreewayDeterministic-v4 --batch-size 4 --max-step 100000 --learning-rate 0.001 --target-network-update 200 --sample-count 10 --env-model EnvModelFreewayManual --vmodel FreewayVNetwork --env-transforms FreewayTransform --save-freq 100 --logdir freeway_td/train20

REM sampleTimer(7,0.2),sampleTimer(13,0.5) -> ikisi icin de p=0 yapildi
REM rm freeway_td/train19 -rf
REM pipenv run python -B oo_td.py FreewayDeterministic-v4 --batch-size 4 --max-step 100000 --learning-rate 0.001 --target-network-update 200 --sample-count 10 --env-model EnvModelFreewayManual --vmodel FreewayVNetwork --env-transforms FreewayTransform --save-freq 100 --logdir freeway_td/train19

REM tavuk variance=20, mean her bir step'de 0'dan 190-20'ye kadar kaydiriliyor (reverse experience replay gibi), network 512x512x512
REM rm freeway_td/train18 -rf
REM pipenv run python -B oo_td.py FreewayDeterministic-v4 --batch-size 4 --max-step 100000 --learning-rate 0.001 --target-network-update 200 --sample-count 10 --env-model EnvModelFreewayManual --vmodel FreewayVNetwork --env-transforms FreewayTransform --save-freq 100 --logdir freeway_td/train18

REM tavuk random'da mean'i shift etmisim, duzeldildi
REM rm freeway_td/train17 -rf
REM pipenv run python -B oo_td.py FreewayDeterministic-v4 --batch-size 4 --max-step 20000 --learning-rate 0.001 --target-network-update 200 --sample-count 50 --env-model EnvModelFreewayManual --vmodel FreewayVNetwork --env-transforms FreewayTransform --save-freq 100 --logdir freeway_td/train17

REM tavuk: random normal, var=50
REM rm freeway_td/train16 -rf
REM pipenv run python -B oo_td.py FreewayDeterministic-v4 --batch-size 4 --max-step 20000 --learning-rate 0.001 --target-network-update 200 --sample-count 50 --env-model EnvModelFreewayManual --vmodel FreewayVNetwork --env-transforms FreewayTransform --save-freq 100 --logdir freeway_td/train16

REM tavuk: random normal, var=80
REM rm freeway_td/train15 -rf
REM pipenv run python -B oo_td.py FreewayDeterministic-v4 --batch-size 4 --max-step 20000 --learning-rate 0.001 --target-network-update 200 --sample-count 50 --env-model EnvModelFreewayManual --vmodel FreewayVNetwork --env-transforms FreewayTransform --save-freq 100 --logdir freeway_td/train15

REM rm freeway_td/train14 -rf
REM pipenv run python -B oo_td.py FreewayDeterministic-v4 --batch-size 4 --max-step 20000 --learning-rate 0.001 --target-network-update 200 --sample-count 50 --env-model EnvModelFreewayManual --vmodel FreewayVNetwork --env-transforms FreewayTransform --save-freq 100 --logdir freeway_td/train14

REM 256x256x256x1
REM rm freeway_td/train13 -rf
REM pipenv run python -B oo_td.py FreewayDeterministic-v4 --batch-size 4 --max-step 20000 --learning-rate 0.001 --target-network-update 200 --sample-count 50 --env-model EnvModelFreewayManual --vmodel FreewayVNetwork --env-transforms FreewayTransform --save-freq 100 --logdir freeway_td/train13

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
