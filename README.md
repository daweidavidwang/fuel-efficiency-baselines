# fuel-efficiency-baselines

conda create -n rl_pcc python=3.9

conda activate rl_pcc

pip install -r requirements.txt

## cyipopt 1.3.0 contains bug on displaying log
conda install -c conda-forge cyipopt==1.2.0


keep background running
screen
screen -ls
screen -r
ctrl + a + d

nohup

TODO:
fuel model
IPOPT based pcc baseline
gym standard gym (s-based)