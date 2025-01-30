# samosa

1. Create and activate a conda environment with Python 3.7 as follows: 
```
conda create -n samis_al python=3.7.16
conda activate samis_al
```
```
conda install pytorch=1.7.1 torchvision=0.8.2 cudatoolkit=11.0
```

**If running on cluster**, do
```
export LD_LIBRARY_PATH=/home/{username}/.conda/envs/cent7/2024.02-py311/samis_al/lib:$LD_LIBRARY_PATH
```
```
pip install -r environment.txt
```

2. Change source code for Dataloader following [here](https://github.com/ningkp/LfOSA/issues/4).
   
4. Make directory results/ and intermediaries/
```
mkdir results
mkdir intermediaries
mkdir new_fair_results
mkdir new_fair_intermediaries
```

5. To run samis_al_high for Cifar10 with 20% mismatch,
```
python samisuk.py --query-strategy samisuk_h --init-percent 1 --known-class 2 --query-batch 1500 --seed 1 --model resnet18 --dataset cifar10 --max-query 11 --max-epoch 300 --stepsize 60 --diversity 1 --gpu 0
```
6. To run samis_al_high for Cifar100 with 20% mismatch,
```
python samisuk.py --query-strategy samisuk_h --init-percent 8 --known-class 20 --query-batch 1500 --seed 1 --model resnet18 --dataset cifar100 --max-query 11 --max-epoch 300 --stepsize 60 --diversity 1 --gpu 0
```

7. To run EOAL for Cifar10 with 20% mismatch,
```
python eoal.py --query-strategy eoal --init-percent 1 --known-class 2 --query-batch 1500 --seed 1 --model resnet18 --dataset cifar10 --max-query 11 --max-epoch 300 --stepsize 60 --diversity 1 --gpu 0
```

8. For Cifar10, init-percent should be 1, known-class should be 2,3,4 for 20% 30% 40% mismatch. For Cifar100 and TinyImagenet, init-percent should be 8, known-class should be 20,30,40 and 40,60,80 respectively.

9. To run lfOSA,
```
python lfosa.py --continue-round -1 --query-strategy lfosa --init-percent 1 --known-class $kc --query-batch 1500 --seed $seed --model resnet18 --dataset cifar10 --max-query 11 --max-epoch 300 --stepsize 60 --diversity 1 --gpu 0 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --weight-cent 0
```

10. To run other baselines,
```
python mqnet.py --query-strategy [mqnet, BADGE, LL, Coreset, Uncertainty_CONF, Uncertainty_Margin] --init-percent 1 --known-class [2,3,4] --query-batch 1500 --seed {} --model ResNet18 --dataset cifar10 --max-query 11 --max-epoch 300 --stepsize 60 --diversity 1 --gpu 0
```

After finishing, for fair evaluation of accuracy,
```
python eval_mqnet.py --query-strategy [mqnet, BADGE, LL, Coreset, Uncertainty_CONF, Uncertainty_Margin] --init-percent 1 --known-class [2,3,4] --query-batch 1500 --seed {} --model ResNet18 --dataset cifar10 --max-query 11 --max-epoch 300 --stepsize 60 --diversity 1 --gpu 0
```
   
11. We are using seeds [1,15,42,85] for our random seeds
