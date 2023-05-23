# drl_wrestling

Participant of the [ICRA 2023 Humanoid Robot Wrestling Competition](https://webots.cloud/run?version=R2023a&url=https://github.com/cyberbotics/wrestling/blob/main/worlds/wrestling.wbt&type=competition).

## Description

This controller employs a policy acquired by the Model-Based Reinforcement Learning (MBRL) algorithm [DreamerV3](https://arxiv.org/abs/2301.04104). All pre-trained models are available for download [here](https://drive.google.com/drive/folders/1i3eUdABo_NtjtT7qd0BoPvuwwoJSwRJZ?usp=share_link).

|  Model  |           Observation Space            |  Action Space   |                                                Revision                                                |
| :-----: | :------------------------------------: | :-------------: | :----------------------------------------------------------------------------------------------------: |
| model03 | Visual (24x24x3) + Proprioceptive (38) | Continuous (14) |                    [main](https://github.com/AndrejOrsula/drl_wrestling/tree/main)                     |
| model02 | Visual (24x24x3) + Proprioceptive (38) | Continuous (15) | [ba448f9](https://github.com/AndrejOrsula/drl_wrestling/tree/ba448f9dc5d309e34e81f3ab0780ff5358c11aeb) |
| model01 | Visual (32x32x3) + Proprioceptive (23) | Continuous (17) | [f5a336d](https://github.com/AndrejOrsula/drl_wrestling/tree/f5a336dd7bdec3d09aceb293981fe23f5b857225) |

## Team

|     Name      |                           Affiliation                           | Occupation  |
| :-----------: | :-------------------------------------------------------------: | :---------: |
| Andrej Orsula | [SpaceR](https://www.spacer.lu), SnT, University of Luxembourg | PhD Student |
