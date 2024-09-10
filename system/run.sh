#!/bin/bash


#SBATCH --job-name=FED
#SBATCH --nodes=1
#SBATCH --nodelist=hpc24
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=train_outs/small/out/%x.%j.out
#SBATCH --error=train_outs/small/errors/%x.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=20010736@st.phenikaa-uni.edu.vn



python .\main.py -data HAR -m harcnn -algo FedAvg -gr 2000 -did 0 -nb 12 -lbs 64 -nc 30 -jr 0.5 -ls 5

python .\main.py -data HAR -m harcnn -algo FedProx -gr 2000 -did 0 -nb 12 -lbs 64 -nc 30 -jr 0.5 -ls 5

python .\main.py -data HAR -m harcnn -algo FedFomo -gr 2000 -did 0 -nb 12 -lbs 64 -nc 30 -jr 0.5 -ls 5

python .\main.py -data HAR -m harcnn -algo FedAMP -gr 2000 -did 0 -nb 12 -lbs 64 -nc 30 -jr 0.5 -ls 5