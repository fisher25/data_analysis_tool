



# C1=C;j=0;  A=zeros(1,n);  For i=1:n;  If sum(C1(i,:))==0; 
# j=j+1;  A(1,j)=i;  End  End  Randpos=[ ];  Tabu=zeros(m,n);  Tabu(:,1)=A(randpos(:))’; 

# C为产品零部件之间的干涉矩阵模型，m为初始蚂蚁的数量，即相当 于产品初始可拆卸零件的个数，n为产品零部件的总个数，Tabu为一数据表，用 其来存储并记录零部件的拆卸序列。
import numpy as np
import random
from matplotlib import pyplot as plt
import pandas as pd
import csv
import xlrd


def is_not_member(a,b):
    for i in a:
        index = np.where(b==i)[0]
        if index.size == 0:
            yield index
        else:
            yield 0
            
# s1=(1,2,3,4,5,6)
# s2=(2,3,5)
# s3=[]
# for i in s1:
# if i not in s2:
#     s3.append(i)

INF = 200
E = 0.001


class Ant_Agorithm:
    def __init__(self) -> None:
        
        self.timecost_matrix = np.array([           
            [E,   8,   7,   4,  INF, 40,  24,  12,  3,  7,  2,   48,  2,   32,  20,  8],
            [2,   E,   9,   4,  8,   40,  24,  12,  3,  9,  6,   48,  22,  32,  20,  8],
            [2,   8,   E,   4,  INF, 40,  24,  12,  3,  7,  6,   48,  40,  32,  20,  8],
            [10,  8,   7,   E,  8,   40,  24,  12,  3,  7,  2,   48,  40,  32,  20,  8],
            [2,   8,   7,   4,  E,   40,  24,  12,  3,  7,  6,   48,  40,  32,  20,  8],
            [10,  60,  15,  4,  8,   E,   INF, INF, 3,  15, 10,  8,   INF, INF, INF, INF],
            [10,  60,  15,  4,  8,   12,  E,   INF, 3,  15, 22,  12,  INF, 8,   INF, INF],
            [10,  60,  15,  4,  8,   12,  14,  E,   3,  15, 26,  34,  10,  22,  8,   INF],
            [2,   8,   7,   4,  8,   26,  26,  12,  E,  7,  34,  42,  18,  30,  12,  8],
            [2,   8,   7,   4,  INF, 34,  34,  20,  3,  E,  6,   50,  26,  38,  20,  16],
            [10,  8,   15,  4,  8,   42,  42,  20,  3,  15, E,   50,  26,  38,  20,  16],
            [10,  8,   15,  4,  8,   42,  INF, INF, 3,  15, 2,   E,   INF, INF, INF, INF],
            [10,  8,   15,  4,  8,   INF, 16,  INF, 3,  15, 34,  24,  E,   INF, INF, INF],
            [10,  8,   15,  4,  8,   16,  4,   INF, 3,  15, 14,  12,  INF, E,   INF, INF],
            [10,  8,   15,  4,  8,    4,  14,  INF, 3,  15, 28,  26,  2,   14,  E,   INF],
            [10,  8,   15,  4,  8,   16,  28,   4,  3,  15, 32,  30,  14,  24,  12,  E]
            ])
        print("shape:",np.size(self.timecost_matrix,0))
        
    def config(self):
        self.ant_cnt = 2
        self.loop_cnt_break = 1000
        self.stroe_info_coe = 2
        self.inspire_info_coe = 2
        self.local_info = 0.1
        self.global_info = 0.5
        self.weight_f1 = 0.5
        self.weight_f2 = 0.5
        self.info_reduce = 0.2                      
        self.node_cnt = np.size(self.timecost_matrix,0)
        self.min_path_cost = 1e10
        self.node_info_pct = np.ones((self.node_cnt,self.node_cnt))     #information precent in each node
        self.node_idx = np.array(range(0,self.node_cnt))
        self.rest_nodes =np.zeros((self.ant_cnt,self.node_cnt),int)
        print("self.rest_nodes",self.rest_nodes)
        self.d_cost = np.divide(1,self.timecost_matrix)
        self.path_ant = np.zeros([self.ant_cnt,self.node_cnt],dtype=int)
        
    def input_info(self):
        filename = './data/data.csv'
        row_num = range(1,7) 
        #[range(7,32),range(61,87),range(107,120)] 
        #  headers=6, usecols=[12,13,14]
        column_names = ['M','N']
        data = self.read_xls()
        print('data:',data)
        return data
        
        
    def read_xls(self):
        data = {}
        df = pd.read_excel('./data/内拆工位.xls',header = None)
        print(df)
        rows =  range(4,13) 
        cols = range(12,21)
        data = np.array(df.loc[rows, cols].values)
        self.step_name = np.array(df.loc[3, cols].values)
        data[data == 'E'] = E
        data[data == 'INF'] = INF
        return data  
        
    def read_csv(self,filename,row_num,column_names):
        data = {}
        with open(filename, 'r',encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            print('reader:',reader)
            headers = next(reader)
            for i, row in enumerate(reader):
                if i  == row_num:
                    for j, name in enumerate(column_names):
                        data[name] = row[headers.index(name)]    
                    break
        return data
        
    
    def ini_path(self):
        
        # self.path_ant = np.zeros([self.ant_cnt,self.node_cnt],dtype=int)
        pos_ini = np.zeros(self.ant_cnt,dtype=int)
        for j in range(self.ant_cnt):
            pos_ini[j] = random.randint(0,self.node_cnt-1)

        self.path_ant[:,0] = pos_ini
        print("path_ant:",self.path_ant)
            
            
    def path_search(self):
        
        for iter in range (self.loop_cnt_break):
            self.ini_path()  
            # rest_node_map = [range(self.node_cnt),range(self.node_cnt)] 
            for k in range(self.ant_cnt):
                
                rest_node = range(self.node_cnt)
                
                for i in range(1,self.node_cnt):
                   
                    cur_node_idx = np.where(rest_node == self.path_ant[k][i-1])[0]
                    cur_node = self.path_ant[k][i-1]
                    rest_node = list(np.delete(rest_node,int(cur_node_idx)))
                    # rest_node_map.append(rest_node)
                    
                    # allow_node = np.take_along_axis(self.node_idx, rest_idx, axis=1)
                    odds = np.zeros(np.size(rest_node))
                    for j in range(len(rest_node)):
                        odds[j] = self.node_info_pct[cur_node][rest_node[j]]**self.stroe_info_coe * self.d_cost[cur_node][rest_node[j]]**self.inspire_info_coe
                        
                    odds_sum = np.divide(odds,np.sum(odds))
                    odds_cum = np.cumsum(odds_sum)
                    print("odds_cum:",odds_cum)

                    
                    target_idx = np.where(odds_cum>0.5)[0]
                    print('target_idx',target_idx)
                    target_idx = np.random.choice(target_idx,1)
                    print('target_idx',target_idx)
                    if len(target_idx) == 0:
                        pass
                    else:
                        target_next = rest_node[target_idx[0]]
                        self.path_ant[k][i] = target_next
            
            LL = np.zeros((self.ant_cnt,1))
            for k in range(self.ant_cnt):
                for i in range(self.node_cnt - 1):
                    LL[k] = LL[k] + self.timecost_matrix[self.path_ant[k][i]][self.path_ant[k][i+1]]
                LL[k] = LL[k] + self.timecost_matrix[self.path_ant[k][self.node_cnt-1]][self.path_ant[k][0]]

            dt = np.zeros((self.node_cnt,self.node_cnt),dtype=int)
            for k in range(self.ant_cnt):
                for i in range(self.node_cnt):
                    print("dt[][]",dt[self.path_ant[k][i]][self.path_ant[k][i]])
                    dt[self.path_ant[k][i]][self.path_ant[k][i]] = dt[self.path_ant[k][i]][self.path_ant[k][i]] + \
                        np.divide(1,LL[k])
                dt[self.path_ant[k][i]][self.path_ant[k][i]] = dt[self.path_ant[k][i]][self.path_ant[k][i]] + \
                        np.divide(1,LL[k])
            self.node_info_pct = (1-self.info_reduce)*self.node_info_pct +dt
            
            min_path_new = np.min(LL) 
            min_path_new_idx = np.argmin(LL)
            if min_path_new < self.min_path_cost:
                self.min_path_cost = min_path_new
                min_path_ret = self.path_ant[min_path_new_idx]
            
            print(self.timecost_matrix)
            print("min_path_ret",min_path_ret)
            print("self.min_path_cost",self.min_path_cost)
        print('self.step_name',self.step_name)
        plt.figure(1)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('dispatch order for minmium time')
        plt.plot(min_path_ret)
        plt.show()
            
            
    
    
    def view(self):
        pass
  
        
    def process(self):
        self.config()
        self.path_search()
        self.view() 
    
    
        
if __name__ == "__main__":
    
    agorithm = Ant_Agorithm()
    agorithm.timecost_matrix = agorithm.input_info()

    agorithm.process()
    
    
    