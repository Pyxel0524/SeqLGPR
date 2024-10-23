# Looking Beneath More: A Sequence-based Localizing Ground Penetrating Radar Framework

Implementation of SeqLGPR in Python, including code for training the model on the GROUNDED dataset.
[![](imgs/video-preview.png)]

## Data

Running this code requires a copy of the Pittsburgh 250k (available [here](https://github.com/Relja/netvlad/issues/42)), 
and the dataset specifications for the Pittsburgh dataset (available [here](https://www.di.ens.fr/willow/research/netvlad/data/netvlad_v100_datasets.tar.gz)).
`pittsburgh.py` contains a hardcoded path to a directory, where the code expects directories `000` to `010` with the various Pittsburth database images, a directory
`queries_real` with subdirectories `000` to `010` with the query images, and a directory `datasets` with the dataset specifications (.mat files).


# Usage

`demo.py` contains the majority of the code, and has three different modes (`train`, `test`, `cluster`) which we'll discuss in mode detail below.


## Paper

"Looking Beneath More: A Sequence-based Localizing Ground Penetrating Radar Framework"

If you use this code, please cite:
```
@INPROCEEDINGS{10610174,
  author={Zhang, Pengyu and Zhi, Shuaifeng and Yuan, Yuelin and Bi, Beizhen and Xin, Qin and Huang, Xiaotao and Shen, Liang},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={Looking Beneath More: A Sequence-based Localizing Ground Penetrating Radar Framework}, 
  year={2024},
  volume={},
  number={},
  pages={8515-8521},
  keywords={Location awareness;Visualization;Ground penetrating radar;Redundancy;Pipelines;Network architecture;Feature extraction},
  doi={10.1109/ICRA57147.2024.10610174}}
```
