REM 2 layer mlp 512x1 (onceki 512x512x512x1 idi
rm freeway_td/test3 -rf
pipenv run python -B oo_td_test.py FreewayDeterministic-v4 --load-weightfile freeway_td/train3/weights_ 3000 100 3300 --env-model EnvModelFreewayManual --vmodel FreewayVNetwork --env-transforms FreewayTransform --enable-render True --render-step 4 --logdir freeway_td/test3
