# [kdd2021_TrafficSiganlControl](http://www.yunqiacademy.org/home/submission)

Basic structure of [DGN](https://github.com/PKU-AI-Edge/DGN) is adopted.

Modifcation trial includes:

1. Customize adjacency matrix  

2. Customize state and reward function

3. [Double Q-learning](https://arxiv.org/abs/1509.06461)

4. Impose low-rank structure of the multiagent Q values (Inpsired by [Sample Efficient Reinforcement Learning via Low-Rank Matrix Estimation](https://arxiv.org/abs/2006.06135))

5. Incorporate futrure state prediction as a auxiliary task (Inspired by [Terminal Prediction as an Auxiliary Task for Deep Reinforcement Learning](https://arxiv.org/abs/1907.10827))

In my experiment, the hyper-parameter (exploration ratio) and the reward function seem to play a dominate role in the performance. The final result is:

![result](https://github.com/Wangjw6/kdd2021_TrafficSiganlControl/blob/main/log/result.png)
 
