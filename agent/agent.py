import pickle
import gym
from DGN import DGN
from  brain import BRAIN
from pathlib import Path
import pickle
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import pandas as pd
# how to import or load local files
import os
import sys
path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(path)
import gym_cfg
with open(path + "/gym_cfg.py", "r") as f:
    pass

class TestAgent():
    def __init__(self):
        np.random.seed(0)
        torch.manual_seed(0)
        self.now_phase = {}
        self.green_sec = 10
        self.red_sec = 5
        self.max_phase = 4
        self.last_change_step = {}
        self.agent_list = []
        self.phase_passablelane = {}
        self.n_ant = 859
        self.n_actions = 8
        self.observation_space = 24 #+ 4 + self.n_actions
        hidden_dim = 400
        self.adj = np.eye(self.n_ant).reshape(1,self.n_ant,self.n_ant) #  np.load(path+'/adj.npy').reshape(1,n_ant,n_ant)
        # self.adj = np.load('./agent/adj.npy').reshape(1,self.n_ant,self.n_ant)
        # self.adj = self.adj/np.sum(self.adj+1e-5,axis=1,keepdims=True)
        self.model = DGN(self.n_ant, self.observation_space, hidden_dim, self.n_actions)
        # self.model = BRAIN(self.n_ant, self.observation_space, hidden_dim, self.n_actions)
        self.model.eval()
        self.last_oboard = 0
        self.last_queue = 0
        try:
            state_dict = torch.load(path+"/save/model164.pth")
            self.model.load_state_dict(state_dict)
        except:
            print('Can not Load')

        df = pd.DataFrame(pd.read_csv(path+'/hash.csv' , index_col=False))
        self.agent_hash = {row[0]: row[1] for row in df.values}

        df = pd.DataFrame(pd.read_csv(path+'/roads.csv'))
        self.roads = {}
        for i in range(df.shape[0]):
            self.roads[int(df.iloc[i, 1])] = [int(df.iloc[i, 2]), int(df.iloc[i, 3]), float(df.iloc[i,4]),float(df.iloc[i, 5]),
                                 float(df.iloc[i, 6]), 0, 0, 0]

        df = pd.DataFrame(pd.read_csv(path+'/signals.csv'))
        self.signals = {}
        for i in range(df.shape[0]):
            self.signals[int(df.iloc[i, 1])] = []
            for j in range(4):
                if int(df.iloc[i, 2 + j]) != -1:
                    self.signals[int(df.iloc[i, 1])].append(self.roads[int(df.iloc[i, 2 + j])][3])
                else:
                    self.signals[int(df.iloc[i, 1])].append(-1)

        self.avail_actions = dict.fromkeys(list(self.agent_hash.keys()), [1 for a in range(4)])
        self.actions = None
        ################################
    def load_model(self,m):
        state_dict = torch.load(path + "/save/model{}.pth".format(m))
        self.model.load_state_dict(state_dict)
        print('Load"%s'%(str(m)))
    # don't modify this function.
    # agent_list is a list of agent_id
    def load_agent_list(self,agent_list):
        self.agent_list = agent_list
        self.now_phase = dict.fromkeys(self.agent_list,1)
        self.last_change_step = dict.fromkeys(self.agent_list,0)
        self.last_action = dict.fromkeys(self.agent_list,[])
    def load_roadnet(self,intersections, roads, agents):
        self.intersections = intersections
        self.roads = roads
        self.agents = agents

    ################################
    def press_adj(self,roads, agent_hash):
        adj = np.zeros([len(agent_hash), len(agent_hash)])
        for k, v in roads.items():
            if v[0] in agent_hash:
                adj[agent_hash[v[0]], agent_hash[v[0]]] += np.sum(v[-3:]) + 1e-5
            if v[0] in agent_hash and v[1] in agent_hash:
                adj[agent_hash[v[0]], agent_hash[v[1]]] = np.sum(v[-3:]) + 1e-5

        adj = adj / np.sum(adj, axis=1, keepdims=True)
        return adj.reshape([1, len(agent_hash), len(agent_hash)])

    def act(self, obs):
        """ !!! MUST BE OVERRIDED !!!
        """
        # here obs contains all of the observations and infos

        # observations is returned 'observation' of env.step()
        # info is returned 'info' of env.step()
        observations = obs['observations']
        info = obs['info']
        actions = {}
        self.n_ant = len(self.agents)
        self.adj = np.eye(self.n_ant).reshape(1,self.n_ant,self.n_ant)
        # preprocess observations
        observations_for_agent = {}
        obs = [[-1] for _ in range(len(self.agent_hash))]
        for key, val in observations.items():
            observations_agent_id = int(key.split('_')[0])
            observations_feature = key[key.find('_') + 1:]
            if (observations_agent_id not in observations_for_agent.keys()):
                observations_for_agent[observations_agent_id] = {}
            val = np.array(val)
            val[val < -1] = 8.

            val = val.tolist()
            now = int(val[0])
            observations_for_agent[observations_agent_id][observations_feature] = val[1:]
        queue = 0
        for k, v in info.items():
            if float(v['speed'][0]) == 0 and len(v['route']) > 1:
                queue += 1

        if now>1600:
            if now %  30 != 0 and self.actions != None:
                return self.actions

        if now <=  1600 and now % 20 != 0 and self.actions != None:

            return self.actions


        for key in observations.keys():
            key = int(key.split('_')[0])
            agent_id = np.zeros([len(self.agent_hash)])
            agent_id[self.agent_hash[key]] = 1
            agent_last_action = np.zeros([self.n_actions])
            agent_last_action[self.now_phase[key]-1] = 1.
            n = self.signals[key].count(-1)
            inet_type = np.zeros([len(self.signals[key])])
            inet_type[n] = 1
            obs[self.agent_hash[key]] =  np.array(observations_for_agent[key]['lane_vehicle_num']).reshape(-1, ).tolist()\
                                 #  +inet_type.tolist() +agent_last_action.tolist()

        q = self.model(torch.Tensor(np.array([obs])), torch.Tensor(self.adj))
        q = q[0]
        # get actions
        # q = torch.svd_lowrank(q, q=2)
        # q = torch.mm(torch.mm(q[0], torch.diag(q[1])), q[2].t())

        for key in self.agent_hash.keys():
            if np.random.rand() <  -0.01:
                a = self.now_phase[key]%8+1
            else:
                action_value = q[self.agent_hash[key]]
                a = action_value.argmax().item()%8 + 1

            self.now_phase[key] = a
            actions[key] = a

        self.actions = actions
        self.last_queue = queue
        self.last_oboard = len(info)
        return actions

scenario_dirs = [
    "test"
]

agent_specs = dict.fromkeys(scenario_dirs, None)
for i, k in enumerate(scenario_dirs):
    # initialize an AgentSpec instance with configuration
    agent_specs[k] = TestAgent()

