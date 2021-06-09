import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
path = 'G:\\base\\kdd2021\\data\\'

class intersection():
    def __init__(self,inter_id,lat,log,is_signalized):
        self.inter_id = inter_id
        self.lat = lat
        self.log = log
        self.is_signalized = is_signalized
class road():
    def __init__(self,from_inter_id, to_inter_id, length, speed_limit, dir1_num_lane, dir2_num_lane,
                 dir1_id, dir2_id,dir1_mov, dir2_mov ):
        self.from_inter_id = from_inter_id
        self.to_inter_id = to_inter_id
        self.length  = length
        self.speed_limit  = speed_limit
        self.dir1_num_lane = dir1_num_lane
        self.dir2_num_lane = dir2_num_lane
        self.dir1_id = dir1_id
        self.dir2_id = dir2_id
        self.dir1_mov = dir1_mov
        self.dir2_mov = dir2_mov

class signal():
    def __init__(self,inter_id,approach_id):
        self.inter_id = inter_id
        self.approach_id = approach_id

def get_geo(f = 'roadnet_round2.txt'):
    path = 'G:\\base\\kdd2021\\data\\'
    _intersections = {}
    _roads = {}
    _signals ={}
    with open(path+f) as data:

        number_of_intersections = 0
        number_of_roads = 0
        number_of_signals = 0

        # init intersections
        csvlines = csv.reader(data, delimiter=' ')
        flag = 0
        for lineNum, line in enumerate(csvlines):
            if len(line)==1 and flag==0:
                number_of_intersections = int(line[0])
                flag = 1
                continue
            if len(line)==1 and flag==1:
                flag = 0
                break
            inter = intersection(inter_id= (line[2]),lat=float(line[0]),log=float(line[1]),is_signalized=int(line[3]))
            _intersections[inter.inter_id] = inter


        number_of_roads = int(line[0])
        # init roads
        flag = 0
        roadid = []
        start_inter = []
        to_inter = []
        start_sig = []
        to_sig = []
        length = []
        inverse = []
        speedlimt = []
        for lineNum, line in enumerate(csvlines):
            if lineNum>=number_of_roads*3:
                break
            if flag==0:
                r = road(from_inter_id=line[0],to_inter_id=line[1],length=float(line[2]),speed_limit=float(line[3]),
                         dir1_num_lane = line[4], dir2_num_lane = line[5],
                         dir1_id = line[6], dir2_id = line[7], dir1_mov = None, dir2_mov = None
                         )
                flag = 1
                roadid.append(int(line[6]))
                start_inter.append(int(line[0]))
                to_inter.append(int(line[1]))
                length.append(float(line[2]))
                inverse.append(int(line[7]))
                speedlimt.append(float(line[3]))

                roadid.append(int(line[7]))
                start_inter.append(int(line[1]))
                to_inter.append(int(line[0]))
                length.append(float(line[2]))
                inverse.append(int(line[6]))
                speedlimt.append(float(line[3]))
                continue
            if flag==1:
                r.dir1_mov = line
                flag = 2
                continue
            if flag==2:
                r.dir2_mov = line
                flag = 0
                _roads[r.from_inter_id+'_'+r.to_inter_id] = r
                continue

        # init signals
                number_of_signals = int(line[0])

        north = [] # road id from north
        east = []
        south = []
        west = []
        type = []
        inter_id = []
        for lineNum, line in enumerate(csvlines):
            sig = signal(inter_id=line[0],approach_id=line[1:])
            inter_id.append(int(line[0]))
            north.append(int(line[1]))
            east.append(int(line[2]))
            south.append(int(line[3]))
            west.append(int(line[4]))
            tag = ''
            for k in range(4):
                if int(line[k+1])>0:
                    tag+='1'
                else:
                    tag+='0'
            type.append(tag)
            _signals[sig.inter_id] = sig
    signals = pd.DataFrame()
    signals['id'] = inter_id
    signals['n'] = north
    signals['e'] = east
    signals['s'] = south
    signals['w'] = west
    signals['t'] = type
    path = 'G:\\base\\kdd2021\\agent'
    signals.to_csv(path + '\\signals.csv')
    roads = pd.DataFrame()
    roads['id'] = roadid
    roads['start_inter'] = start_inter
    roads['to_inter'] = to_inter
    roads['length'] = length
    roads['inverse'] = inverse
    roads['speedlimt']=speedlimt
    path = 'G:\\base\\kdd2021\\agent'
    roads.to_csv(path+'\\roads.csv')
    return _intersections, _roads, _signals

def analyze_connect(_intersections, _roads, _signals):
    path =  'G:\\base\\kdd2021\\agent'
    p = path + '\\hash.csv'
    df = pd.DataFrame(pd.read_csv(p, index_col=False))
    agent_hash = {row[0]: row[1] for row in df.values}
    adj = np.eye( len(agent_hash) )
    adj = np.zeros([len(agent_hash),len(agent_hash)])
    for key,value in _roads.items():
        from_ = int(value.from_inter_id)
        to_ = int(value.to_inter_id)
        if (from_ in agent_hash) and (to_ in agent_hash):
            pass
        else:
            continue
        adj[agent_hash[to_],agent_hash[from_]] = 1.
        adj[agent_hash[from_], agent_hash[to_]] = 1.
    # adj = adj / np.sum(adj + 1e-5, axis=1, keepdims=True)
    np.fill_diagonal(adj, -np.sum(adj, axis=1, keepdims=True))
    print(adj)
    adj = -adj
    # adj = adj.reshape([1,len(agent_hash),len(agent_hash)])
    plt.imshow(adj)
    plt.show()
    np.save(path+'\\adj.npy',adj)

def get_flow(f = 'G:\\mcgill\\useful\\flow_round2.txt'):
    with open( f) as data:
        csvlines = csv.reader(data, delimiter=' ')
        flow = np.zeros([3600,3600])
        for lineNum, line in enumerate(csvlines):
            if (lineNum+2)%3==0:
                flow[int(line[0]),int(line[1])] += ((int(line[1])-int(line[0]))/int(line[2]))
    print(np.sum(flow))
    for i in range(flow.shape[0]):
        if np.sum(flow[i])!=0:
            print(i)
    plt.plot(np.sum(flow,axis=1),'-o')
    plt.show()
if __name__ == '__main__':
    # get_flow()
    _intersections, _roads, _signals = get_geo()

    # for key,value in _intersections.items():
    #     if value.is_signalized==1:
    #         plt.scatter(value.lat,value.log,color='blue')
    #     else:
    #         plt.scatter(value.lat,value.log,color='red')
    # # plt.show()
    #
    # for key,value in _roads.items():
    #     plt.plot([_intersections[value.from_inter_id].lat,_intersections[value.to_inter_id].lat],
    #              [_intersections[value.from_inter_id].log,_intersections[value.to_inter_id].log],color='black')
    # plt.show()

    analyze_connect(_intersections, _roads, _signals)