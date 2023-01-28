# deepflex-experiments

Using AMI ID `ami-0b7e0d9b36f4e8f14`:
`Deep Learning AMI GPU PyTorch 1.13.1 (Ubuntu 20.04) 20230103`

Commands to run on AWS:

```sh
source activate pytorch
git clone <this repo>
cd deepflex-experiments
bash download.sh

# example of NCCL w/ 10 epochs
python example_mod.py -a resnet34 --epochs 10 "test"  --dist-url 'tcp://127.0.0.1:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0

# example of GLOO w/ 10 epochs
python example_mod.py -a resnet34 --epochs 10 "test"  --dist-url 'tcp://127.0.0.1:FREEPORT' --dist-backend 'gloo' --multiprocessing-distributed --world-size 1 --rank 0
```
