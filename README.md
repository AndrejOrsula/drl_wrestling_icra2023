# drl_wrestling

Participant of the [ICRA 2023 Humanoid Robot Wrestling Competition](https://webots.cloud/run?version=R2023a&url=https://github.com/cyberbotics/wrestling/blob/main/worlds/wrestling.wbt&type=competition).

## Description

This controller employs a policy trained via Model-Based Reinforcement Learning (MBRL). Details about the policy for specific models can be found below. All models are available for download [here](https://drive.google.com/drive/folders/1i3eUdABo_NtjtT7qd0BoPvuwwoJSwRJZ?usp=share_link).

|  Model  |                      Algorithm                       |           Observation Space            |  Action Space   | Size of Policy (Actor) |
| :-----: | :--------------------------------------------------: | :------------------------------------: | :-------------: | :--------------------: |
| model01 | [DreamerV3](https://arxiv.org/abs/2301.04104) (MBRL) | Visual (32x32x3) + Proprioceptive (23) | Continuous (17) |   1330210 parameters   |

## Team

|     Name      |                           Affiliation                           | Occupation  |
| :-----------: | :-------------------------------------------------------------: | :---------: |
| Andrej Orsula | [SpaceR](https://www.spacer.lu/), SnT, University of Luxembourg | PhD Student |
