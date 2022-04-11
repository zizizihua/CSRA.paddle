# CSRA.Paddle
An implementation of [CSRA](https://arxiv.org/abs/2108.02456) based on [paddlepaddle](https://www.paddlepaddle.org.cn/)

# Installation
1. Install PaddleCls
```bash
git clone https://github.com/PaddlePaddle/PaddleClas.git
pip install -r requirements.txt
pip install -v -e .
```
2. Make dir 'projects' under PaddleCls path, and copy CSRA.paddle to the 'projects' folder

# Prepare VOC2007 dataset
1. download voc2007 train/val and test data to './data'
2. convert xml annotation to txt annotation
```bash
cd PaddleCls/projects/CSRA.paddle/
python tools/convert_dataset.py \
    --data_root ./data/VOCdevkit/VOC2007 \
    --out_file ./data/VOCdevkit/VOC2007/train_list.txt \
    --mode trainval

python tools/convert_dataset.py \
    --data_root ./data/VOCdevkit/VOC2007 \
    --out_file ./data/VOCdevkit/VOC2007/test.txt \
    --mode test
```

# Train
```bash
cd PaddleCls/projects/CSRA.paddle/
python tools/train.py -c configs/ResNet101_CSRA_voc.yaml
```

# Eval
```bash
cd PaddleCls/projects/CSRA.paddle/
python tools/eval.py -c configs/ResNet101_CSRA_voc.yaml \
    -o Arch.pretrained="output/ResNet_CSRA/epoch_28"
```

# Result on VOC2007
ResNet101, 1 head, lam 0.1
|  Acc  |  mAP  |
|  ---  |  ---  |
| 98.69 | 94.60 |
