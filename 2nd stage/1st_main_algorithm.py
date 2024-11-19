import itertools
import torch.nn as nn
from sac import SAC
from replay_memory import ReplayMemory
import time
import numpy as np
from Orbit_Dynamics.CW_Prop import CW_Prop
import numpy as np
from gymnasium import spaces
import math
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

class test_env:
    def __init__(self):
         #状态空间设置为无限连续状态空间，虽然不知道相比设成离散空间有什么影响
        self.attack_action_space=spaces.Box(low=np.array([-1.0,-0.1,-1.0]), high=np.array([1.0,0.1,1.0]), shape=(3,), dtype=np.float32)  #降低收敛难度，设限
        self.defense_action_space=spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32) 
        self.observation_space=spaces.Box(-np.inf,np.inf,shape=(13,),dtype=np.float32)
        self.target_distance=1e4
        self.velocity_tolerence=3
        self.omega=2*math.pi/(24*3600)
        self.GM=3.986e14
        self.radius=(self.GM/self.omega**2)**(1/3) #42241080.06788323
        self.i=0
        self.info=0
        self.t=3600.0
        self.gamma = 0.2   # 范数限制 γ,脉冲大小限制
        self.tau=0.2 #要超过85.9%

    def target_generate(self,seed): #生成能让仅测角安心工作的目标位置
        np.random.seed(seed)
        original_target_arc=np.random.uniform(-0.00214285714285714285714285714286,-0.00261904761904761904761904761905)
        original_target_radius=self.radius+np.random.uniform(-2e4,-1e4)
        original_target_pos=np.array([original_target_radius*math.sin(original_target_arc),
                                np.random.uniform(-5,5)*1000,
                                self.radius-original_target_radius*math.cos(original_target_arc)])
        original_target_abs_vel=(self.GM/original_target_radius)**(1/2)
        ref_vel=(self.GM/self.radius)**(1/2)
        vel_noise=np.random.uniform(-0.05,0.05)
        original_target_vel=np.array([original_target_abs_vel*math.cos(original_target_arc)+self.omega*original_target_pos[2]-ref_vel+vel_noise,
                                 0.0,#np.random.uniform(-0.0, 0.0),
                                 original_target_abs_vel*math.sin(original_target_arc)-self.omega*original_target_pos[0]+vel_noise]) #
        original_red_pos=np.array([0,0,0])
        original_red_vel=np.array([0,0,0])
        relative_pos=original_red_pos-original_target_pos
        relative_vel=original_red_vel-original_target_vel
        return relative_pos,relative_vel

    def reset(self,seed=None): #,prop_t
        if seed is None:
            seed=self.i
        np.random.seed(seed)
        super().__init__()
        self.Xt=[]
        self.Vt=[]
        self.Xc=[]
        self.Vc=[]
        self.Ut=[]
        self.Uc=[]
        self.Tv=[]
        self.Tpos=[]
        self.target_relative_pos,self.target_relative_vel=self.target_generate(seed) #必须重新设一下seed，不然会影响到主函数
        self.info=0
        self.blue_arc=np.random.uniform(-0.0082857,-0.0106531)
        self.blue_radius=self.radius+np.random.uniform(-1e5,1e5)
        self.blue_pos=np.array([self.blue_radius*math.sin(self.blue_arc),
                                np.random.uniform(-5,5)*1000,
                                self.radius-self.blue_radius*math.cos(self.blue_arc)])
        self.blue_abs_vel=(self.GM/self.blue_radius)**(1/2)
        self.ref_vel=(self.GM/self.radius)**(1/2)
        self.vel_noise=np.array([np.random.uniform(-3,3),
                                0,
                                np.random.uniform(-3,3)])
        self.blue_vel=np.array([self.blue_abs_vel*math.cos(self.blue_arc)+self.omega*self.blue_pos[2]-self.ref_vel, #+self.vel_noise
                                 0.0,#np.random.uniform(-0.0, 0.0),
                                 self.blue_abs_vel*math.sin(self.blue_arc)-self.omega*self.blue_pos[0]]) #+self.vel_noise
        self.red_arc=np.random.uniform(-0.00023673637094339374,0.00023673637094339374)
        self.red_pos=np.array([self.radius*math.sin(self.red_arc),
                                np.random.uniform(-2,2)*1000,
                                self.radius-self.radius*math.cos(self.red_arc)])
        self.red_vel=np.array([self.ref_vel*math.cos(self.red_arc)-self.ref_vel+self.omega*self.red_pos[2]+self.vel_noise[0],
                                 np.random.uniform(-0.05, 0.05),
                                 self.ref_vel*math.sin(self.red_arc)-self.omega*self.red_pos[2]+self.vel_noise[2]])
        self.Xt.append(self.red_pos)
        self.Vt.append(self.red_vel)
        self.Xc.append(self.blue_pos)
        self.Vc.append(self.blue_vel)
        # self.red_pos_list.append(self.red_pos)
        # self.time_delay=np.random.normal(loc=self.delay_mu, scale=self.delay_sigma, size=1)
        self.done=False
        self.fuel=np.array([30])
        self.target_pos=self.red_pos-self.target_relative_pos
        self.target_vel=self.red_vel-self.target_relative_vel
        self.Tpos.append(self.target_pos)
        self.Tv.append(self.target_vel)
        state=np.concatenate((self.blue_pos/5e4,self.blue_vel,self.target_pos/5e4,self.target_vel,self.fuel))
        self.IQ=np.array([1.0,1.0,1.0]) #目标智慧度
        self.count_step=0 #步数
        return state
    
    def opt_escape(self,Xt,Vt,Xc,Vc): #最优的逃逸方式
        Xt=np.array([[Xt[0]],[Xt[1]],[Xt[2]]]) #向量转置
        Vt=np.array([[Vt[0]],[Vt[1]],[Vt[2]]])
        Xc=np.array([[Xc[0]],[Xc[1]],[Xc[2]]])
        Vc=np.array([[Vc[0]],[Vc[1]],[Vc[2]]])
        mat_A=np.array([[1,0,6*(self.omega*self.t-math.sin(self.omega*self.t))],
                        [0,math.cos(self.omega*self.t),0],
                        [0,0,4-3*math.cos(self.omega*self.t)]])
        mat_B=np.array([[(4/self.omega*math.sin(self.omega*self.t)-3*self.t),0,2/self.omega*(1-math.cos(self.omega*self.t))],
                        [0,1/self.omega*math.sin(self.omega*self.t),0],
                        [2/self.omega*(-1+math.cos(self.omega*self.t)),0,1/self.omega*math.sin(self.omega*self.t)]])
        # 计算向量 C
        mat_C=mat_A@Xt+mat_B@Vt-mat_A@Xc-mat_B@Vc

        # 定义目标函数，用于求解 ||u1|| - gamma = 0
        def objective(lambda_):
            # 构造矩阵 (B^T B - lambda I)
            matrix=mat_B.T@mat_B-lambda_*np.eye(mat_B.shape[1]) 
            # 考虑矩阵是否可逆
            if np.linalg.cond(matrix) > 1 / np.finfo(matrix.dtype).eps:
                # 如果矩阵接近奇异，返回一个大值以避免数值问题
                return np.inf  
            # 解线性方程组 (B^T B - lambda I) u1 = -B^T C
            u1=-np.linalg.solve(matrix,-mat_B.T@mat_C)
            
            # 返回 ||u1|| - gamma
            return np.linalg.norm(u1)-self.gamma
        # 使用 fsolve 寻找合适的 lambda
        lambda_initial_guess = 1

        # fsolve 需要目标函数接近0
        #fsolve是一种基于牛顿法的根寻找算法，适用于求解一般的非线性方程，总之就是求一个lambda使objective=0，也即|u1|=gamma
        lambda_solution, = fsolve(objective, lambda_initial_guess) 

        # 计算最终的 u1
        matrix_opt = mat_B.T @ mat_B - lambda_solution * np.eye(mat_B.shape[1])
        u1_optimal = -np.linalg.solve(matrix_opt, -mat_B.T@mat_C)
        u1_optimal=np.ndarray.flatten(u1_optimal.T)
        return(u1_optimal)
    
    def pos_escape(self,Xt,Vt,Xc,Vc): #沿着距离反向喷脉冲
        ut=(Xt-Xc).T/np.linalg.norm(Xt-Xc)*self.gamma
        ut=np.ndarray.flatten(ut.T)
        return(ut)

    def step(self,blue_action):#,red_action
        self.count_step+=1
        self.Uc.append(blue_action) #有count_step个
        '''【设定目标逃逸策略】'''
        # red_impulse=self.opt_escape(self.red_pos,self.red_vel,self.blue_pos,self.blue_vel) #最优跑
        # red_impulse=self.pos_escape(self.red_pos,self.red_vel,self.blue_pos,self.blue_vel) #位置反向跑
        # red_impulse=self.gamma/np.linalg.norm(self.blue_vel)*self.blue_vel #速度同向跑
        # red_impulse=np.array([0,0,0]) #不跑
        alpha=np.random.normal(0,2*math.pi)
        beta=np.random.normal(0,2*math.pi)
        red_impulse=np.array([self.gamma*math.cos(alpha)*math.cos(beta),self.gamma*math.sin(beta),self.gamma*math.sin(alpha)*math.cos(beta)]) #乱跑
        print(math.degrees(math.acos(np.linalg.norm(self.opt_escape(self.red_pos,self.red_vel,self.blue_pos,self.blue_vel)*red_impulse)/(np.linalg.norm(self.opt_escape(self.red_pos,self.red_vel,self.blue_pos,self.blue_vel))*np.linalg.norm(red_impulse)))))
        self.after_blue_vel=self.blue_vel+blue_action
        self.next_blue_pos,self.next_blue_vel=CW_Prop(self.blue_pos,self.after_blue_vel,self.omega,3600)
        if self.count_step<2:
            # guess_red_pos,guess_red_vel=CW_Prop(self.red_pos,self.red_vel,self.omega, self.t) #直接外推预测，不走DMM
            guess_red_impulse=self.opt_escape(self.red_pos,self.red_vel,self.blue_pos,self.after_blue_vel) #配合DMM用的,推测目标在s0采用最优逃逸策略
            guess_red_vel=self.red_vel+guess_red_impulse
            guess_red_pos,guess_red_vel=CW_Prop(self.red_pos,guess_red_vel,self.omega,3600) #推测目标目前状态

            self.next_target_pos=guess_red_pos-self.target_relative_pos
            self.next_target_vel=guess_red_vel-self.target_relative_vel 
        self.Xc.append(self.next_blue_pos) #有count_step+1个
        self.Vc.append(self.next_blue_vel)
        #self.new_red_pos,self.new_red_vel=CW_Prop(self.red_pos,self.red_vel,self.omega,3600)
        self.Ut.append(red_impulse)
        self.red_vel=self.red_vel+red_impulse
        self.next_red_pos,self.next_red_vel=CW_Prop(self.red_pos,self.red_vel,self.omega,3600)
        self.Xt.append(self.next_red_pos) #有count_step+1个       
        self.Vt.append(self.next_red_vel)
        if self.count_step>=2:
            # guess_red_pos,guess_red_vel=CW_Prop(self.Xt[-3],self.Vt[-3],self.omega, 2*self.t) #直接外推
            best_u1=self.opt_escape(self.Xt[-3],self.Vt[-3],self.Xc[-3],self.Vc[-3])
            guess_red_pos_now,guess_red_vel_now=CW_Prop(self.Xt[-3],self.Vt[-3]+best_u1,self.omega,self.t)
            best_u2=self.opt_escape(guess_red_pos_now,guess_red_vel_now,self.Xc[-2],self.Vc[-2])
            guess_red_pos,guess_red_vel=CW_Prop(guess_red_pos_now,guess_red_vel_now+best_u2,self.omega,self.t)

            self.next_target_pos=guess_red_pos-self.target_relative_pos
            self.next_target_vel=guess_red_vel-self.target_relative_vel  
            # elf.next_target_pos,self.next_target_vel=self.DMM(self.Xt,self.Vt,self.Ut,self.Xc,self.Vc) 
        self.fuel=np.array([self.fuel[0]-np.linalg.norm(blue_action)])
        state=np.concatenate((self.next_blue_pos/5e4,self.next_blue_vel,self.next_target_pos/5e4,self.next_target_vel,self.fuel))
        self.next_distance=np.linalg.norm(self.next_blue_pos-self.next_target_pos)
        self.next_velocity_difference=np.linalg.norm(self.next_target_vel-self.next_blue_vel)
        self.blue_pos=self.next_blue_pos
        self.blue_vel=self.next_blue_vel
        self.red_pos=self.next_red_pos
        self.red_vel=self.next_red_vel
        self.Tpos.append(self.next_target_pos)
        self.Tv.append(self.next_target_vel)

        if self.fuel[0]<0 or (self.next_distance<self.target_distance and self.next_velocity_difference<self.velocity_tolerence): # and vel_goal
            self.done=True
        if self.done:
            if self.fuel[0]<0:
                self.info=0 #用来区分越界和到达目标
            elif (self.next_distance<self.target_distance) and self.fuel[0]>=0:
                self.info=1
            self.i+=1

        return state, self.done, self.info, self.fuel[0], self.next_distance, self.next_velocity_difference,self.Xt,self.Xc,self.Vt,self.Vc,self.Tpos,self.Tv

# 字典形式存储全部参数
args={'policy':"Gaussian", # Policy Type: Gaussian | Deterministic (default: Gaussian)
        'eval':True, # Evaluates a policy a policy every 10 episode (default: True)
        'gamma':0.99, # discount factor for reward (default: 0.99)
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
        'task':'Plot', # 测试或训练或画图，Train,Test,Plot
        'activation':nn.ReLU, #激活函数类型
        'plot_type':'3D-1line', #'3D-1line'为三维图，一条曲线；'2D-2line'为二维图，两条曲线
        'plot_title':'test.svg',
        'max_episodes':1e6, #测试算法（eval=False）情况下的总步数
        'evaluate_freq':100, #训练过程中每多少个epoch之后进行测试
        'seed':20000323, #网络初始化的时候用的随机数种子  
        'max_epoch':100000,
        'logs':True} #是否留存训练参数供tensorboard分析 

time_start=time.time()
# Environment
env = test_env()
# Agent
agent = SAC(env.observation_space.shape[0], env.attack_action_space, args) #discrete不能用shape，要用n提取维度数量
agent.load_checkpoint('Zsofarsogood_16_25.111777437874117.pt')
time_start=time.time()
episodes = 1
done_num=0
avg_fuel=0
avg_steps=0
for i  in range(episodes):
    state = env.reset(323)
    episode_reward = 0
    done = False
    steps = 0
    fuel_consumption=0
    while not done:
        action = agent.select_action(state, evaluate=True) #evaluate为True时为确定性网络，直接输出mean
        fuel_consumption=fuel_consumption+np.linalg.norm(action[0:3])
        next_state, done, info, fuel, distance, velo,Xt,Xc,Vt,Vc,Tpos,Tv = env.step(action)
        state = next_state
        steps+=1
        if info:
            done_num+=1
        if done:
            if args['task']=='Plot':
                pos_x_target = np.array([array[0] for array in Xt])
                pos_y_target = np.array([array[1] for array in Xt])
                pos_z_target = np.array([array[2] for array in Xt])
                pos_x_chaser = np.array([array[0] for array in Xc])
                pos_y_chaser = np.array([array[1] for array in Xc])
                pos_z_chaser = np.array([array[2] for array in Xc])
                velo_x_target = np.array([array[0] for array in Vt])
                velo_y_target = np.array([array[1] for array in Vt])
                velo_z_target = np.array([array[2] for array in Vt])
                velo_x_chaser = np.array([array[0] for array in Vc])
                velo_y_chaser = np.array([array[1] for array in Vc])
                velo_z_chaser = np.array([array[2] for array in Vc])
                pos_x_obj = np.array([array[0] for array in Tpos])
                pos_y_obj = np.array([array[1] for array in Tpos])
                pos_z_obj = np.array([array[2] for array in Tpos])
                velo_x_obj = np.array([array[0] for array in Tv])
                velo_y_obj = np.array([array[1] for array in Tv])
                velo_z_obj = np.array([array[2] for array in Tv])
                i_seq=np.array([i for i in range(len(Xt))])
                
                font = {'family': 'serif',
                'serif': 'Times New Roman',
                'weight': 'normal',
                'size': 15,
                }
                plt.rc('font', **font)
                plt.style.use('seaborn-whitegrid')
                fig = plt.figure(figsize=(17,6))
                ax1 = fig.add_subplot(131, projection='3d') #, projection='3d' #生成3d子图
                ax2 = fig.add_subplot(132) #, projection='3d'
                ax3 = fig.add_subplot(133)
                #Axes对象是图形的绘图区域，可以在其上进行各种绘图操作。
                #通过gca()函数可以获取当前图形的Axes对象，如果该对象不存在，则会创建一个新的。
                #projection='3d' 参数指定了Axes对象的投影方式为3D，即创建一个三维坐标系。
                ax1.plot(pos_x_target/1e3,pos_y_target/1e3,pos_z_target/1e3,'r',linewidth=1, label="Target")
                ax1.plot(pos_x_obj/1e3,pos_y_obj/1e3,pos_z_obj/1e3,'b',linewidth=1, label="Objective")
                ax1.plot(pos_x_chaser/1e3,pos_y_chaser/1e3,pos_z_chaser/1e3,'g',linewidth=1, label="Pursuer")
                ax2.plot(i_seq*3600,(pos_x_obj-pos_x_chaser)/1e3,'r',linewidth=1, label="x distance") 
                ax2.plot(i_seq*3600,(pos_y_obj-pos_y_chaser)/1e3,'b',linewidth=1, label="y distance") 
                ax2.plot(i_seq*3600,(pos_z_obj-pos_z_chaser)/1e3,'g',linewidth=1, label="z distance")
                ax3.plot(i_seq*3600,velo_x_obj-velo_x_chaser,'r',linewidth=1, label="x velocity") 
                ax3.plot(i_seq*3600,velo_y_obj-velo_y_chaser,'b',linewidth=1, label="y velocity") 
                ax3.plot(i_seq*3600,velo_z_obj-velo_z_chaser,'g',linewidth=1, label="z velocity")
                #ax.scatter(data_x,data_y,data_z,'b',s=1)
                ax1.set_xlabel('x/km', fontsize=15)
                ax1.set_ylabel('y/km', fontsize=15)
                ax1.set_zlabel('z/km', fontsize=15)
                ax1.xaxis.labelpad = 20  # 负值将标签向图像外部移动
                ax1.yaxis.labelpad = 20  # 负值将标签向图像外部移动
                ax1.zaxis.labelpad = 10  # 负值将标签向图像外部移动
                # 获取当前的x轴和y轴标签对象
                zlabel = ax1.get_zlabel()
                # 设置z轴标签的方向，使其与z轴平行
                ax1.zaxis.label.set_rotation(0)  # 0度表示与X轴平行
                ax2.set_xlabel('t/s', fontsize=15)
                ax2.set_ylabel('distance/km', fontsize=15)
                # 获取当前的x轴和y轴
                ax2_xaxis = ax2.get_xaxis()
                ax2_yaxis = ax2.get_yaxis()
                # 调整x轴标签的位置到x轴的最右端
                ax2_xaxis.label.set_horizontalalignment('right')
                ax2_yaxis.label.set_position((1, 0))
                # 调整y轴标签的位置到y轴的最上端
                ax2_yaxis.label.set_verticalalignment('top')
                y_axis_pos = ax2.get_ylim()[1]  # 获取y轴的最上端位置
                ax2_yaxis.label.set_position((0, 1))
                # 将x轴标签移动到x轴的末端,调节y坐标控制其上下移动
                ax2.xaxis.set_label_coords(1.05, -0.05)
                # 将y轴标签移动到y轴的末端,调节x坐标控制其左右移动
                ax2.yaxis.set_label_coords(-0.15, 0.9)
                ax3.set_xlabel('t/s', fontsize=15)
                ax3.set_ylabel('velocity/(m/s)', fontsize=15)
                ax1.legend(fontsize=15,loc='upper right')
                ax2.legend(fontsize=15)
                ax3.legend(fontsize=15)
                # ax.set_xlim(np.min([np.min(data_x),np.min(data_y),np.min(data_z)]),np.max([np.max(data_x),np.max(data_y),np.max(data_z)]))
                # ax.set_ylim(np.min([np.min(data_x),np.min(data_y),np.min(data_z)]),np.max([np.max(data_x),np.max(data_y),np.max(data_z)]))
                # ax.set_zlim(np.min([np.min(data_x),np.min(data_y),np.min(data_z)]),np.max([np.max(data_x),np.max(data_y),np.max(data_z)]))
                plt.tight_layout()# 调整布局使得图像不溢出
                plt.savefig(args['plot_title'], format='svg', bbox_inches='tight')# 'logs/{}epoch-{}steps.png'.format(epoch,steps))
                plt.show()
                break
            else:
                break
    avg_steps+=steps
    if info:
        avg_fuel+=fuel_consumption
avg_steps/= episodes
avg_fuel /= done_num
#writer.add_scalar('avg_reward/test', avg_reward, i_episode)
time_end=time.time()
print("----------------------------------------")
print("Test Episodes: {},done num:{},fuel consumption：{},average steps:{}".format(episodes,done_num,avg_fuel,avg_steps))
print("----------------------------------------")
print((time_end-time_start)/episodes)