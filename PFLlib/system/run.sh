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

# har data 
python main.py -data HAR -m harcnn -algo FedAvg -gr 2000 -did 0 -nc 30 -lbs 64 -nb 6 -jr 0.4 -ls 5

python main.py -data HAR -m harcnn -algo FedProx -gr 2000 -did 0 -nc 30 -lbs 64 -nb 6 -jr 0.4 -ls 5

python main.py -data HAR -m harcnn -algo FedFomo -gr 2000 -did 0 -nc 30 -lbs 64 -nb 6 -jr 0.4 -ls 5 

python main.py -data HAR -m harcnn -algo MOON -gr 2000 -did 0 -nc 30 -lbs 64 -nb 6 -jr 0.4 -ls 5

python main.py -data HAR -m harcnn -algo FedGen -gr 2000 -did 0 -nc 30 -lbs 64 -nb 6 -jr 0.4 -ls 5

python main.py -data HAR -m harcnn -algo FedDistill -gr 2000 -did 0 -nc 30 -lbs 64 -nb 6 -jr 0.4 -ls 5

# sleep data - ts-transformer
python main.py -data SLEEP -m transformer -algo FedAvg -gr 2000 -did 0 -nc 12 -lbs 64 -nb 12  -jr 0.4 -ls 5

python main.py -data SLEEP -m transformer -algo FedProx -gr 2000 -did 0 -nc 12 -lbs 64 -nb 12  -jr 0.4 -ls 5

python main.py -data SLEEP -m transformer -algo FedFomo -gr 2000 -did 0 -nc 12 -lbs 64 -nb 12  -jr 0.4 -ls 5

python main.py -data SLEEP -m transformer -algo MOON -gr 2000 -did 0 -nc 12 -lbs 64 -nb 12  -jr 0.4 -ls 5

python main.py -data SLEEP -m transformer -algo FedGen -gr 2000 -did 0 -nc 12 -lbs 64 -nb 12  -jr 0.4 -ls 5

python main.py -data SLEEP -m transformer -algo FedDistill -gr 2000 -did 0 -nc 12 -lbs 64 -nb 12  -jr 0.4 -ls 5

# sleep data - 2d-cnn

python main.py -data SLEEP -m harcnn -algo FedProx -gr 2000 -did 0 -nc 12 -lbs 64 -nb 12 -jr 0.4 -ls 5

python main.py -data SLEEP -m harcnn -algo FedFomo -gr 2000 -did 0 -nc 12 -lbs 64 -nb 12 -jr 0.4 -ls 5

python main.py -data SLEEP -m harcnn -algo MOON -gr 2000 -did 0 -nc 12 -lbs 64 -nb 12 -jr 0.4 -ls 5

python main.py -data SLEEP -m harcnn -algo FedGen -gr 2000 -did 0 -nc 12 -lbs 64 -nb 12 -jr 0.4 -ls 5

python main.py -data SLEEP -m harcnn -algo FedDistill -gr 2000 -did 0 -nc 12 -lbs 64 -nb 12 -jr 0.4 -ls 5

python main.py -data SLEEP -m harcnn -algo FedMTL -gr 2000 -did 0 -nc 12 -lbs 64 -nb 12 -jr 0.4 -ls 5
