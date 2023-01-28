# deepflex-experiments

On AWS:

```
source activate pytorch
git clone <this repo>
cd deepflex-experiments
bash download.sh

# example of NCCL w/ 10 epochs
python example_mod.py -a resnet34 --epochs 10 “test”  --dist-url 'tcp://127.0.0.1:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0

# example of GLOO w/ 10 epochs
python example_mod.py -a resnet34 --epochs 10 “test”  --dist-url 'tcp://127.0.0.1:FREEPORT' --dist-backend 'gloo' --multiprocessing-distributed --world-size 1 --rank 0
```