# DRL-For-imbalanced-Classification

This repository provides an altered version of the code for the DQNimb model proposed in paper [DRL for imbalanced classification](https://arxiv.org/abs/1901.01379?context=cs.LG):

* **train_model**: Training the agent with DQN algorithm.
* **ICMDP_Env**: The simulated environment for imabalanced classification.
* **get_model**: Define the network structure for image, text or structured.
* **get_data**: Loading the balanced datasets and constructing the imbalanced datasets according to the imbalanced rate.

![image.png](https://i.loli.net/2019/11/26/4pr2qK5VQoBhNj1.png)
![table.png](https://i.loli.net/2019/11/26/iAkLw7JlsXFu56g.png)

## Requirements

* See `requirements.txt`
* Latest version of [Keras-RL](https://github.com/keras-rl/keras-rl.git) (pip version does not include all callbacks)
* CUDA 10.0
* cuDNN 7.4

## Quick Start

```bash
python train_model.py --model image --data famnist --imb-rate 0.04 --min-class 456 --maj-class 789 --training-steps 120000
python train_model.py --model image --data mnist --imb-rate 0.04 --min-class 2 --maj-class 013456789 --training-steps 71000
python train_model.py --model structured --data credit --imb-rate 0.01 --min-class 1 --maj-class 0 --training-steps 500000
python train_model.py --model text --data imdb --imb-rate 0.1 --min-class 0 --maj-class 1 --training-steps 150000
```

## Tensorboard command

```bash
tensorboard --logdir logs/fit
```
