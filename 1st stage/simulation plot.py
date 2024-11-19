import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
import os
import math
from Orbit_Dynamics.CW_Prop import CW_Prop
import datetime
import itertools
import torch.nn as nn
from sac import SAC
from replay_memory import ReplayMemory
import env
import time

class env:
    def __init__(self):
         #状态空间设置为无限连续状态空间，虽然不知道相比设成离散空间有什么影响
        self.attack_action_space=spaces.Box(low=np.array([-2.0,-2.0,-2.0,600]), high=np.array([2.0,2.0,2.0,1200]), shape=(4,), dtype=np.float32)  #降低收敛难度，设限
        self.defense_action_space=spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32) 
        self.observation_space=spaces.Box(-np.inf,np.inf,shape=(13,),dtype=np.float32)
        self.blue_pos=np.array([np.random.uniform(250, 400)*1000,
                                np.random.uniform(-50, 50)*1000,
                                np.random.uniform(-100, 100)*1000])
        # self.red_pos=np.array([np.random.uniform(-50, 50)*1000,
        #                         np.random.uniform(-50, 50)*1000,
        #                         np.random.uniform(-20, 20)*1000])
        self.red_pos=np.array([0,0,0])
        self.distance=np.linalg.norm(self.blue_pos-self.red_pos)
        self.blue_vel= np.random.uniform(-10, 10, size=(3,))
        #self.red_vel= np.random.uniform(-5, 5, size=(3,))
        self.red_vel=np.array([0,0,0])
        self.target_distance=5000
        self.fuel=np.array([20])
        self.omega=2*math.pi/(24*3600)
        self.i=0
        self.info=0
        
    def angle_observe(self,blue_pos,blue_vel,red_pos,red_vel,time_interval): 
        blue_pos,blue_vel=CW_Prop(blue_pos,blue_vel,self.omega,time_interval)
        red_pos,red_vel=CW_Prop(red_pos,red_vel,self.omega,time_interval)
        relative_pos=red_pos-blue_pos
        x_angle=math.degrees(math.acos(relative_pos[0]/np.linalg.norm(relative_pos))) #这里的三行就是把姿态转换矩阵拆了
        y_angle=math.degrees(math.acos(relative_pos[1]/np.linalg.norm(relative_pos)))
        z_angle=math.degrees(math.acos(relative_pos[2]/np.linalg.norm(relative_pos)))
        return x_angle,z_angle

    def reset(self): #,prop_t
        np.random.seed(self.i)
        super().__init__()
        self.blue_pos_list=[]
        self.red_pos_list=[]
        self.info=0
        self.blue_pos=np.array([np.random.uniform(-105, -95)*1000,
                                 np.random.uniform(-5, 5)*1000,
                                 np.random.uniform(-20, -10)*1000])
        # self.red_pos=np.array([np.random.uniform(-5, 5)*1000,
        #                         np.random.uniform(0, 0)*1000,
        #                         np.random.uniform(-5, 5)*1000])
        self.blue_vel=np.array([np.random.uniform(1, 3),
                                 np.random.uniform(0, 0),
                                 np.random.uniform(1, 3)])
        # self.red_vel= np.random.uniform(-1, 1, size=(3,))
        # self.blue_pos=np.array([-300000,0,-15000])
        self.red_pos=np.array([0,0,0])
        self.red_vel=np.array([0,0,0])
        # self.blue_vel= np.array([10,0,10])
        # self.blue_pos_list.append(self.blue_pos)
        # self.red_pos_list.append(self.red_pos)
        self.done=False
        self.fuel=np.array([50])
        self.init_distance=np.linalg.norm((self.blue_pos-self.red_pos))
        state=[]
        o1,o2=self.angle_observe(self.blue_pos,self.blue_vel,self.red_pos,self.red_vel,-300)
        state.append(o1)
        state.append(o2)
        o3,o4=self.angle_observe(self.blue_pos,self.blue_vel,self.red_pos,self.red_vel,-150)
        state.append(o3)
        state.append(o4)
        o5,o6=self.angle_observe(self.blue_pos,self.blue_vel,self.red_pos,self.red_vel,0)
        state.append(o5)
        state.append(o6)
        state.append(0)
        state.append(0)
        state.append(0)
        o7,o8=self.angle_observe(self.blue_pos,self.blue_vel,self.red_pos,self.red_vel,150)
        state.append(o7)
        state.append(o8)
        o9,o10=self.angle_observe(self.blue_pos,self.blue_vel,self.red_pos,self.red_vel,300)
        state.append(o9)
        state.append(o10)
        state=np.array(state)
        return state

    def step(self,blue_action):#,red_action
        state=[]
        o1,o2=self.angle_observe(self.blue_pos,self.blue_vel,self.red_pos,self.red_vel,-300)
        state.append(o1)
        state.append(o2)
        o3,o4=self.angle_observe(self.blue_pos,self.blue_vel,self.red_pos,self.red_vel,-150)
        state.append(o3)
        state.append(o4)
        o5,o6=self.angle_observe(self.blue_pos,self.blue_vel,self.red_pos,self.red_vel,0)
        state.append(o5)
        state.append(o6)
        state.append(blue_action[0])
        state.append(blue_action[1])
        state.append(blue_action[2])
        natural_blue_pos,_=CW_Prop(self.blue_pos,self.blue_vel,self.omega,blue_action[3])
        natural_red_pos,_=CW_Prop(self.red_pos,self.red_vel,self.omega,blue_action[3])
        self.blue_vel=self.blue_vel+blue_action[0:3]
        manueuver_blue_pos,_=CW_Prop(self.blue_pos,self.blue_vel,self.omega,300)
        manueuver_red_pos,_=CW_Prop(self.red_pos,self.red_vel,self.omega,300)
        o7,o8=self.angle_observe(self.blue_pos,self.blue_vel,self.red_pos,self.red_vel,150)
        state.append(o7)
        state.append(o8)
        o9,o10=self.angle_observe(self.blue_pos,self.blue_vel,self.red_pos,self.red_vel,300)
        state.append(o9)
        state.append(o10)
        state=np.array(state)
        next_blue_pos,self.blue_vel=CW_Prop(self.blue_pos,self.blue_vel,self.omega,blue_action[3])
        next_red_pos,self.red_vel=CW_Prop(self.red_pos,self.red_vel,self.omega,blue_action[3])
        self.blue_pos_list.append(next_blue_pos)
        self.blue_pos=next_blue_pos
        self.red_pos=next_red_pos
        state=np.array(state)
        self.fuel=np.array([self.fuel[0]-np.linalg.norm(blue_action[0:3])])
        #self.distance=np.linalg.norm(self.blue_pos-self.red_pos)

        if self.fuel[0]<0 or np.linalg.norm(next_blue_pos-next_red_pos)<=self.target_distance:
            self.done=True
        
        if not self.done:
            # natural_relative=natural_red_pos-natural_blue_pos
            # manueuver_relative=manueuver_red_pos-manueuver_blue_pos
            # obs_buff=np.dot(natural_relative,manueuver_relative)/(np.linalg.norm(natural_relative)*np.linalg.norm(manueuver_relative))
            # if obs_buff>0:
            #     obs_buff=5-obs_buff*5
            # else:
            #     obs_buff=-10
            reward=(np.linalg.norm(natural_blue_pos-natural_red_pos)-np.linalg.norm(next_blue_pos-next_red_pos))/1000 #+obs_buff

        if self.done:
            if self.fuel[0]<0:
                self.info=0 #用来区分越界和到达目标
                reward=0
            elif np.linalg.norm(next_blue_pos-next_red_pos)<=self.target_distance and self.fuel[0]>=0:
                self.info=1
                reward=400
            self.i+=1

        return state, reward, self.done, self.info

    def plot(self, args, data_x, data_y, data_z=None):
        if data_z!=None and args['plot_type']=="3D-1line":
            fig = plt.figure()
            ax = fig.gca(projection='3d') #Axes对象是图形的绘图区域，可以在其上进行各种绘图操作。
                                        #通过gca()函数可以获取当前图形的Axes对象，如果该对象不存在，则会创建一个新的。
                                        #projection='3d' 参数指定了Axes对象的投影方式为3D，即创建一个三维坐标系。
            plt.plot(0,0,0,'r*') #画一个位于原点的星形
            plt.plot(data_x,data_y,data_z,'b',linewidth=1) #画三维图
            ax.set_xlabel('x', fontsize=15)
            ax.set_ylabel('y', fontsize=15)
            ax.set_zlabel('z', fontsize=15)
            ax.set_xlim(-10,10)
            ax.set_ylim(-10,10)
            ax.set_zlim(-10,10)
            if not os.path.exists('logs'):
                os.makedirs('logs')
            plt.savefig(args['plot_title'])# 'logs/{}epoch-{}steps.png'.format(epoch,steps)

args={'policy':"Gaussian", # Policy Type: Gaussian | Deterministic (default: Gaussian)
        'eval':True, # Evaluates a policy a policy every 10 episode (default: True)
        'gamma':0.975, # discount factor for reward (default: 0.99)
        'tau':0.2, # target smoothing coefficient(τ) (default: 0.005) 参数tau定义了目标网络软更新时的平滑系数，
                     # 它控制了新目标值在更新目标网络时与旧目标值的融合程度。
                     # 较小的tau值会导致目标网络变化较慢，从而增加了训练的稳定性，但也可能降低学习速度。
        'lr':0.0003, # learning rate (default: 0.0003)
        'alpha':0.2, # Temperature parameter α determines the relative importance of the entropy\term against the reward (default: 0.2)
        'automatic_entropy_tuning':True, # Automaically adjust α (default: False)
        'batch_size':512, # batch size (default: 256)
        'num_steps':1000, # maximum number of steps (default: 1000000)
        'hidden_sizes':[512,256,128], # 隐藏层大小，带有激活函数的隐藏层层数等于这一列表大小
        'updates_per_step':1, # model updates per simulator step (default: 1) 每步对参数更新的次数
        'start_steps':1000, # Steps sampling random actions (default: 10000) 在开始训练之前完全随机地进行动作以收集数据
        'target_update_interval':10, # Value target update per no. of updates per step (default: 1) 目标网络更新的间隔
        'replay_size':10000000, # size of replay buffer (default: 10000000)
        'cuda':True, # run on CUDA (default: False)
        'LOAD PARA':False, #是否读取参数
        'task':'Train', # 测试或训练或画图，Train,Test,Plot
        'activation':nn.ReLU, #激活函数类型
        'plot_type':'2D-2line', #'3D-1line'为三维图，一条曲线；'2D-2line'为二维图，两条曲线
        'plot_title':'reward-steps.png',
        'seed':20000323, #网络初始化的时候用的随机数种子  
        'max_epoch':50000,
        'logs':True} #是否留存训练参数供tensorboard分析 

if args['logs']==True:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter('./runs/')

# Environment
env = env.env()
# Agent
agent = SAC(env.observation_space.shape[0], env.attack_action_space, args) #discrete不能用shape，要用n提取维度数量
#Tensorboard
'''创建一个SummaryWriter对象，用于将训练过程中的日志信息写入到TensorBoard中进行可视化。
   SummaryWriter()这是创建SummaryWriter对象的语句。SummaryWriter是TensorBoard的一个API，用于将日志信息写入到TensorBoard中。
   format括号里内容是一个字符串格式化的表达式，用于生成一个唯一的日志目录路径。{}是占位符，format()方法会将占位符替换为对应的值
   datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")：这是获取当前日期和时间，并将其格式化为字符串。
   strftime()方法用于将日期时间对象转换为指定格式的字符串。在这里，日期时间格式为"%Y-%m-%d_%H-%M-%S"，表示年-月-日_小时-分钟-秒。
   "autotune" if args.automatic_entropy_tuning else ""：这是一个条件表达式，用于根据args.automatic_entropy_tuning的值来决定是否插入"autotune"到日志目录路径中。
   'runs/{}_SAC_{}_{}_{}'是一个字符串模板，其中包含了四个占位符 {}。当使用 format() 方法时，传入的参数会按顺序替换占位符，生成一个新的字符串。'''
#显示图像：用cmd（不是vscode的终端） cd到具体存放日志的文件夹（runs），然后tensorboard --logdir=./
#或者直接在import的地方点那个启动会话
#如果还是不行的话用netstat -ano | findstr "6006" 在cmd里查一下6006端口有没有占用，用taskkill全杀了之后再tensorboard一下

# Memory
memory = ReplayMemory(args['replay_size'])

if args['task']=='Train':
    # Training Loop
    updates = 0
    best_avg_reward=0
    total_numsteps=0
    steps_list=[]
    episode_reward_list=[]
    avg_reward_list=[]
    if args['LOAD PARA']==True:
        agent.load_checkpoint("sofarsogood.pt")
        best_avg_reward=50
    for i_episode in itertools.count(1): #itertools.count(1)用于创建一个无限迭代器。它会生成一个连续的整数序列，从1开始，每次递增1。
        success=False
        episode_reward = 0
        done=False
        episode_steps = 0
        state = env.reset()
        while True:
            # if args['start_steps'] > total_numsteps:
            #     action = env.attack_action_space.sample()   # 开始训练前随机动作若干次获取数据
            # else:
            action = agent.select_action(state)  # 开始输出actor网络动作
            if len(memory) > args['batch_size']:
                # Number of updates per step in environment 每次交互之后可以进行多次训练...
                for i in range(args['updates_per_step']):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args['batch_size'], updates)
                    if args['logs']==True:
                        writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                        writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                        writer.add_scalar('loss/policy', policy_loss, updates)
                        writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                        writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    updates += 1

            next_state, reward, done, info= env.step(action) # Step
            episode_steps += 1
            episode_reward += reward #没用gamma是因为在sac里求q的时候用了
            total_numsteps+=1
            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            memory.push(state, action, reward, next_state, done) # Append transition to memory
            state = next_state

            if done:
                #env.plot(i_episode,episode_steps)
                steps_list.append(i_episode)
                episode_reward_list.append(episode_reward)
                if len(episode_reward_list)>=500:
                    avg_reward_list.append(sum(episode_reward_list[-500:])/500)
                else:
                    avg_reward_list.append(sum(episode_reward_list)/len(episode_reward_list))
                if info:
                    success=True
                break
        if args['logs']==True:
            writer.add_scalar('reward/train', episode_reward, i_episode)
        print("Episode: {}, episode steps: {}, reward: {}, succeed: {}".format(i_episode, episode_steps, round(episode_reward, 4), success))
        # round(episode_reward,2) 对episode_reward进行四舍五入，并保留两位小数

        if i_episode % 200 == 0 and args['eval'] is True: #评价上一个训练过程
            avg_reward = 0.
            episodes = 20
            if args['LOAD PARA']==True:
                episodes=50
            done_num=0
            for _  in range(episodes):
                state = env.reset()
                episode_reward = 0
                done = False
                steps=0
                while not done:
                    action = agent.select_action(state, evaluate=True) #evaluate为True时为确定性网络，直接输出mean
                    next_state, reward, done, info = env.step(action)
                    episode_reward += reward
                    state = next_state
                    steps+=1
                    if info:
                        done_num+=1
                    if done:
                        break
                avg_reward += episode_reward
            avg_reward /= episodes
            if args['logs']==True:
                writer.add_scalar('avg_reward/test', avg_reward, i_episode)
            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {},完成数：{}".format(episodes, round(avg_reward, 4),done_num))
            print("----------------------------------------")
            if avg_reward>best_avg_reward:
                best_avg_reward=avg_reward
                agent.save_checkpoint('sofarsogood.pt')
            if done_num==episodes:
                agent.save_checkpoint("approach-no-impulse.pt")
                env.plot(args, steps_list, episode_reward_list, avg_reward_list)
                break
        if i_episode==args['max_epoch']:
            print("训练失败，{}次仍未完成训练".format(args['max_epoch']))
            env.plot(args, steps_list, episode_reward_list, avg_reward_list)
            if args['logs']==True:
                writer.close()
            break

if args['task']=='Test':
    agent.load_checkpoint('approach-no-impulse.pt')
    avg_reward = 0
    episodes = 100
    done_num=0
    for i  in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        steps=0
        while not done:
            action = agent.select_action(state, evaluate=True) #evaluate为True时为确定性网络，直接输出mean
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
            steps+=1
            if info:
                done_num+=1
            if done:
                break
        avg_reward += episode_reward
    avg_reward /= episodes
    #writer.add_scalar('avg_reward/test', avg_reward, i_episode)
    print("----------------------------------------")
    print("Test Episodes: {}, Avg. Reward: {},完成数：{}".format(episodes, round(avg_reward, 4),done_num))
    print("----------------------------------------")

if args['task']=='Plot':
    agent.load_checkpoint('approach-no-impulse.pt')
    done_num=0
    x_list=[]
    y_list=[]
    z_list=[]
    for i  in range(episodes):
        state = env.reset()
        done = False
        steps=0
        while not done:
            action = agent.select_action(state, evaluate=True) #evaluate为True时为确定性网络，直接输出mean
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
            steps+=1
            if info:
                done_num+=1
            if done:
                break
        avg_reward += episode_reward
    avg_reward /= episodes
    #writer.add_scalar('avg_reward/test', avg_reward, i_episode)
    print("----------------------------------------")
    print("Test Episodes: {}, Avg. Reward: {},完成数：{}".format(episodes, round(avg_reward, 4),done_num))
    print("----------------------------------------")