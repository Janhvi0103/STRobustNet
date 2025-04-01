## STRobustNet: Efficient Change Detection via Spatial-Temporal Robust Representations in Remote Sensing (TGRS 2025) 

<img src="figures\method.png">

[Link to paper](https://ieeexplore.ieee.org/abstract/document/10879578)  

## Usage
## Requiements

```
Python 3.8
pytorch 1.7.1
torchvision 0.8.2
einops 0.8.1
cuda 11.0
```
- Please see `requirements.txt` for all the other requirements.

## Datasets

[LEVIR-CD](https://justchenhao.github.io/LEVIR/)\
[WHU-CD](https://gpcv.whu.edu.cn/data/building_dataset.html)\
[SYSU-CD](https://github.com/liumency/SYSU-CD)

### Data structure

```
Change detection data set with pixel-level binary labels；
├─A
├─B
├─label
└─list
```

`A`: images of t1 phase;

`B`: images of t2 phase;

`label`: label maps;

`list`: contains `train.txt, val.txt or test.txt`, each file records the image names (XXX.png) in the change detection dataset.

## Train
We have placed the training scripts in the scripts folder. You can run a command like
```bash
sh ./scripts/run_LEVIR.sh
``` 
to train the model.

## Evaluation
We have placed the evaluation scripts in the scripts folder. You can run a command like
```bash
sh ./scripts/eval.sh
``` 
to evaluate the model.

## Pretrained Model
Coming soon

## Citation
If you use this code for your research, please cite our paper:

```bibtex
@ARTICLE{10879578,
  author={Zhang, Hong and Teng, Yuhang and Li, Haojie and Wang, Zhihui},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={STRobustNet: Efficient Change Detection via Spatial–Temporal Robust Representations in Remote Sensing}, 
  year={2025},
  volume={63},
  number={},
  pages={1-15},
  doi={10.1109/TGRS.2025.3540794}}
```
## Acknowledgements
We would like to extend our sincere appreciation to the authors of the following projects for making their code available, which we have utilized in our work: [BIT](https://github.com/justchenhao/BIT_CD), [Changeformer](https://github.com/wgcban/ChangeFormer)