import CBEngine
import gym
import agent.gym_cfg as gym_cfg
from agent.mybuff import ReplayBuffer
from agent.brain import BRAIN
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np
import pandas as pd
from evaluate import run_simulation
from pathlib import Path
import os
import sys
import subprocess
import time
np.random.seed(0)
torch.manual_seed(0)
n_ant = 859
n_actions = 8
observation_space = 48 #+ 4 + n_actions
hidden_dim = 800

GAMMA = 0.99

i_episode = 0
capacity = 200000
batch_size = 128
n_epoch = 4
epsilon = 0.8
score = 0
scores = 99999
buff = ReplayBuffer(capacity)
model = BRAIN(n_ant, observation_space, hidden_dim, n_actions)
model_tar = BRAIN(n_ant, observation_space, hidden_dim, n_actions)
optimizer = optim.Adam(model.parameters(), lr=0.002)#,betas=0.9)
path = "./agent/save/model%s.pth" % (str(i_episode))
torch.save(model.state_dict(), path)
# optimizer = optim.RMSprop(model.parameters(), lr=0.000025)
# optimizer = optim.SGD(model.parameters(),lr=0.0001)
if False:
    path = "./agent/save/model.pth"
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)

model_tar.load_state_dict(model.state_dict())
O = np.ones((batch_size, n_ant, observation_space))
Next_O = np.ones((batch_size, n_ant, observation_space))
Matrix = np.ones((batch_size, n_ant, n_ant))
Next_Matrix = np.ones((batch_size, n_ant, n_ant))

simulator_cfg_file = './cfg/simulator.cfg'
mx_step = 180
gym_cfg_instance = gym_cfg.gym_cfg()
epoch = 500
# gym
env = gym.make(
    'CBEngine-v0',
    simulator_cfg_file=simulator_cfg_file,
    thread_num=1,
    gym_dict=gym_cfg_instance.cfg,
    metric_period=200
)

adj = np.eye(n_ant).reshape(1, n_ant, n_ant)  # is good
reward_adj = np.load('./agent/adj.npy').reshape(n_ant, n_ant)

reward_set = []
loss_set = []
queue_set = []
observations, info = env.reset()
k = 0
agent_hash = {}

for key, val in observations.items():
    observations_agent_id = int(key.split('_')[0])
    if observations_agent_id not in agent_hash:
        agent_hash[observations_agent_id] = k
        k += 1
df = pd.DataFrame.from_dict(agent_hash.items(), orient="columns")
df.to_csv('./agent/hash.csv', index=None)

df = pd.DataFrame(pd.read_csv('./agent/roads.csv'))
roads = {}
for i in range(df.shape[0]):
    roads[int(df.iloc[i, 1])] = [int(df.iloc[i, 2]), int(df.iloc[i, 3]), float(df.iloc[i,4]),float(df.iloc[i, 5]),
                                 float(df.iloc[i, 6]), 0, 0, 0]
roads_ = roads.copy()
df = pd.DataFrame(pd.read_csv('./agent/signals.csv'))
signals = {}
for i in range(df.shape[0]):
    signals[int(df.iloc[i, 1])] = []
    for j in range(4):
        if int(df.iloc[i, 2 + j]) != -1:
            signals[int(df.iloc[i, 1])].append(roads[int(df.iloc[i, 2 + j])][3])
        else:
            signals[int(df.iloc[i, 1])].append(-1)

signal_available_lanes = dict.fromkeys(agent_hash.keys(), [])
for k in agent_hash.keys():
    for l in range(4):
        if signals[k][l] != -1:
            signal_available_lanes[k] += [3 * l, 3 * l + 1, 3 * l + 2]

vehicles = {}
road_signal_hash={}
for agent in agent_hash.keys():
    for k, v in roads.items():
        if  v[3] in signals[agent]:
        # if k in signals[agent] or v[3] in signals[agent]:
            if agent in road_signal_hash:
                road_signal_hash[k].append(agent)
            else:
                road_signal_hash[k] = [agent]
signal_hash = {'200': 1, '000': 1, '010': 2, '210': 2, '300': 3, '100': 3, '310': 4, '210': 4,
                               '001': 5, '011': 5, '101': 6, '110': 6, '201': 7, '211': 7, '301': 8, '311': 8, }
# agent_roads_pressure_hash = {}
# for agent in agent_hash.keys():
#     agent_roads_pressure_hash[agent] = [[] for _ in range(8)]
#     for k, v in  roads.items():
#         if v[1] == agent:
#             approach = signals[agent].index(k)
#             for l in range(3):
#                 sg = str(approach) + str(l)
#                 if sg in signal_hash:
#                     # pressure[signal_hash[sg] - 1] += float(v[-3 + l])
#                     agent_roads_pressure_hash[agent][signal_hash[sg] - 1].append([k,l])
def load_agent_submission():
    def resolve_dirs(root_path: str, input_dir: str = None, output_dir: str = None):
        root_path = Path(root_path)

        if input_dir is not None:
            input_dir = Path(input_dir)
            output_dir = Path(output_dir)

            # path in cfg must be consistent with middle_output_dir
            submission_dir = input_dir
            scores_dir = output_dir

        else:
            raise ValueError('need input dir')

        if not scores_dir.exists():
            os.makedirs(scores_dir)

        output_path = scores_dir / "scores.txt"

        return submission_dir, scores_dir

    # find agent.py
    module_path = None
    cfg_path = None
    submission_dir, scores_dir = resolve_dirs(
        os.path.dirname(__file__), 'agent', 'out'
    )
    for dirpath, dirnames, file_names in os.walk(submission_dir):
        for file_name in [f for f in file_names if f.endswith(".py")]:
            if file_name == "agent.py":
                module_path = dirpath

            if file_name == "gym_cfg.py":
                cfg_path = dirpath

    assert (
            module_path is not None
    ), "Cannot find file named agent.py, please check your submission zip"
    assert (
            cfg_path is not None
    ), "Cannot find file named gym_cfg.py, please check your submission zip"
    sys.path.append(str(module_path))

    # This will fail w/ an import error of the submissions directory does not exist
    from agent import gym_cfg
    from agent import agent

    gym_cfg_instance = gym_cfg.gym_cfg()

    return agent.agent_specs, gym_cfg_instance


# construct adjancency based on pressure
def press_adj(roads, agent_hash):
    adj = np.zeros([len(agent_hash), len(agent_hash)])
    for k, v in roads.items():
        if v[0] in agent_hash:
            adj[agent_hash[v[0]], agent_hash[v[0]]] += np.sum(v[-3:]) + 1e-5
        if v[0] in agent_hash and v[1] in agent_hash:
            adj[agent_hash[v[0]], agent_hash[v[1]]] = np.sum(v[-3:]) + 1e-5

    adj = adj / np.sum(adj, axis=1, keepdims=True)
    return adj.reshape([1, len(agent_hash), len(agent_hash)])


if __name__ == '__main__':

    while i_episode < epoch:
        ################################# Initialize #########################################################
        i_episode += 1
        steps = 0
        score = 0
        last_avg_speed = 0

        env.set_warning(False)
        env.set_log(False)
        observations, info = env.reset()
        observations_for_agent = {}
        now_phase = dict.fromkeys(list(agent_hash.keys()),3)
        agent_pressure = dict.fromkeys(list(agent_hash.keys()), 1)
        obs = [[-1] for _ in range(len(agent_hash))]
        f1 = 'lane_vehicle_num'
        f2 = 'lane_speed'
        for key, val in observations.items():
            observations_agent_id = int(key.split('_')[0])
            observations_feature = key[key.find('_') + 1:]
            if (observations_agent_id not in observations_for_agent.keys()):
                observations_for_agent[observations_agent_id] = {}

            val = np.array(val)
            val[val < -1] = 8.

            val = val.tolist()
            observations_for_agent[observations_agent_id][observations_feature] = val[1:]


        for key in agent_hash.keys():
            agent_id = np.zeros([len(agent_hash)])
            agent_id[agent_hash[key]] = 1
            agent_last_action = np.zeros([n_actions])
            agent_last_action[now_phase[key] - 1] = 1.
            n = signals[key].count(-1)
            inet_type = np.zeros([len(signals[key])])
            inet_type[n] = 1
            obs[agent_hash[key]] = np.array(observations_for_agent[key]['lane_speed']).reshape(-1, ).tolist() \
                                   + np.array(observations_for_agent[key]['lane_vehicle_num']).reshape(-1, ).tolist()
                                  # +inet_type.tolist() +agent_last_action.tolist()


        last_vehicle_num = len(info)
        wait_times = {}
        last_action = dict.fromkeys(list(agent_hash.keys()), [])

        #################################### Interact ###############################################
        begin = time.time()
        while steps < mx_step:

            action = [-1 for _ in range(n_ant)]  # for training
            action_ = {}  # for implemented action
            action_rl = []
            q,_ = model(torch.Tensor(np.array([obs])), torch.Tensor(adj))
            q = q[0]
            for key in agent_hash.keys():
                action_rl.append(q[agent_hash[key]].argmax().item() + 1)
                if np.random.rand() < epsilon:
                    a =  now_phase[key]%8 + 1
                else:
                    a = q[agent_hash[key]].argmax().item() + 1
                action_[key] = a
                action[agent_hash[key]] = a - 1
                last_action[key].append(a)
                now_phase[key] = a
            reward = [0. for _ in range(n_ant)]

            for _ in range(2):
                next_observations, rwd, dones, info = env.step(action_)
                steps += 1

                if steps % 20 == 0:
                    print('RL ', np.var(action_rl))
                    print('AL ', np.var(list(action_.values())))

                observations_for_agent = {}
                next_obs = [[-1] for _ in range(len(agent_hash))]
                for key, val in next_observations.items():
                    observations_agent_id = int(key.split('_')[0])
                    observations_feature = key[key.find('_') + 1:]
                    if (observations_agent_id not in observations_for_agent.keys()):
                        observations_for_agent[observations_agent_id] = {}
                    val = np.array(val)
                    val[val<-1]=8.
                    val = val.tolist()
                    observations_for_agent[observations_agent_id][observations_feature] = val[1:]
                for key in agent_hash.keys():
                    next_obs[agent_hash[key]] =  np.array(observations_for_agent[key]['lane_speed']).reshape(-1, ).tolist() \
                                   + np.array(observations_for_agent[key]['lane_vehicle_num']).reshape(-1, ).tolist()

                queue = 0
                vehicles = {}

                for k, v in info.items():
                    lane = 4 + (int(v['drivable'][0]) - 100 * int(v['road'][0]))
                    try:
                        agents = road_signal_hash[int(v['road'][0])]
                    except:
                        continue

                    if float(v['speed'][0]) ==0.  and len(v['route']) > 1:
                        # roads[int(v['road'][0])][lane] += 1.  # / roads[int(v['road'][0])][2]
                        queue += 1
                        # for agent in agents:
                        #     stop[agent_hash[agent]] +=1
                    else:
                        for agent in agents:
                            # print(roads[int(v['road'][0])][2])
                            reward[agent_hash[agent]] += 1. / 8.#/(roads[int(v['road'][0])][2]/1000)
                            # move[agent_hash[agent]] +=1


            print('Episode %s, Steps %s, Queue %d, Onroad %d, Reward %s ' % (
            str(i_episode), str(steps), queue, len(info), str(np.mean(reward) )))

            buff.add(np.array(obs), action, reward, np.array(next_obs), adj, adj, 0)
            obs = next_obs
            score += (np.mean(reward))


        print('Time cost{} min'.format((time.time()-begin)/60.))
        print('Num of experience',buff.num_experiences)
        ######################################## Record ##################################################

        queue_set.append(queue)
        reward_set.append(score)
        if buff.num_experiences < 1000:
            continue
        epsilon = epsilon * 0.9
        if epsilon < 0.02:
            epsilon = 0.02

        print('*' * 9, 'CUMULATIVE reward', score, '*' * 9, 'epsilon', epsilon)


        ########################################## Train ################################################
        for e in range(n_epoch):
            batch = buff.getBatch(batch_size)
            for j in range(batch_size):
                sample = batch[j]
                O[j] = sample[0]
                Next_O[j] = sample[3]
                Matrix[j] = sample[4]
                Next_Matrix[j] = sample[5]

            q_values,predictions = model(torch.Tensor(O), torch.Tensor(Matrix))

            # target_q_values = model_tar(torch.Tensor(Next_O), torch.Tensor(Next_Matrix)).max(dim=2)[0]
            # target_q_values = np.array(target_q_values.cpu().data)

            # Double Q setting
            target_action_indices,_ = model(torch.Tensor(Next_O) , torch.Tensor(Next_Matrix) )
            target_action_indices = target_action_indices.max(dim=2)[1]
            target_q_values,_ = model_tar(torch.Tensor(Next_O) , torch.Tensor(Next_Matrix) )
            target_q_values = target_q_values.gather(2, target_action_indices.view(-1,n_ant,1)).detach()
            target_q_values = np.array(target_q_values.cpu().data)

            expected_q = np.array(q_values.cpu().data)

            for j in range(batch_size):
                sample = batch[j]
                for i in range(n_ant):
                    expected_q[j][i][sample[1][i]] = sample[2][i] + (1 - sample[6]) * GAMMA * target_q_values[j][i]


            loss1 = (q_values - torch.Tensor(expected_q).detach()).pow(2).mean()
            loss2 =  0.01*(predictions-torch.Tensor(Next_O)).pow(2).mean()
            loss = loss1+loss2
            print(loss1.data.numpy(),loss2.data.numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_set.append(loss.data.numpy())

        if i_episode > 60:
            path = "./agent/save/model%s.pth" % (str(i_episode))
            torch.save(model.state_dict(), path)
        if i_episode>100:
            print('Evaluate...')
            subprocess.run(
                ["python3", "evaluate.py", "--input_dir=agent", "--output_dir=out", "--sim_cfg=cfg/simulator.cfg",
                 "--m={}".format(i_episode)])


        def soft_update(net_target, net, tau=0.02):
            for target_param, param in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


        # model_tar.load_state_dict(model.state_dict())
        soft_update(model_tar, model, tau=0.02)
        res = pd.DataFrame()
        res['cumulative reward'] = reward_set
        res.to_csv('train.csv')
        res = pd.DataFrame()
        res['qloss'] = loss_set
        res.to_csv('loss.csv')
        qs = pd.DataFrame()
        qs['queue'] = queue_set
        qs.to_csv('queue.csv')

