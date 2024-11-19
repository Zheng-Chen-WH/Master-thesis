import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
import os
import math
from Orbit_Dynamics.CW_Prop import CW_Prop

'''每次都会忘的vsc快捷键：
    打开设置脚本：ctrl+shift+P
    多行注释：ctrl+/
    关闭多行注释：选中之后再来一次ctrl+/
    多行缩进：tab
    关闭多行缩进：选中多行shift+tab'''

class env:
    def __init__(self):
         #状态空间设置为无限连续状态空间，虽然不知道相比设成离散空间有什么影响
        self.attack_action_space=spaces.Box(low=np.array([-1.0,-0.1,-1.0]), high=np.array([1.0,0.1,1.0]), shape=(3,), dtype=np.float32)  #降低收敛难度，设限
        self.defense_action_space=spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32) 
        self.observation_space=spaces.Box(-np.inf,np.inf,shape=(13,),dtype=np.float32)
        self.target_distance=5e3
        self.velocity_tolerence=2
        self.omega=2*math.pi/(24*3600)
        self.GM=3.986e14
        self.radius=(self.GM/self.omega**2)**(1/3) #42241080.06788323
        self.i=0
        self.info=0
        # self.delay_mu=3600
        # self.delay_sigma=0.5

    def target_generate(self):
        np.random.seed(self.i)
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
    
    def state_generate(self,pos,vel,seed=None):
        if seed is not None:
            np.random.seed(seed)
        impulse_random=np.array([np.random.uniform(-0.5,0.5),0,np.random.uniform(-0.5,0.5)])
        new_vel=vel+impulse_random
        new_pos,new_vel=CW_Prop(pos,vel,self.omega,3600)
        return new_pos, new_vel

    def reset(self,seed=None): #,prop_t
        if seed is None:
            np.random.seed(self.i)
        else:
            np.random.seed(seed)
        super().__init__()
        self.blue_pos_list=[]
        #self.red_pos_list=[]
        self.target_relative_pos,self.target_relative_vel=self.target_generate()
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
        self.blue_pos_list.append(self.blue_pos)
        # self.red_pos_list.append(self.red_pos)
        # self.time_delay=np.random.normal(loc=self.delay_mu, scale=self.delay_sigma, size=1)
        self.done=False
        self.new_done=False
        self.fuel=np.array([30])
        self.target_pos=self.red_pos-self.target_relative_pos
        self.target_vel=self.red_vel-self.target_relative_vel
        state=np.concatenate((self.blue_pos/5e4,self.blue_vel,self.target_pos/5e4,self.target_vel,self.fuel))
        return state

    def step(self,blue_action):#,red_action
        self.blue_vel=self.blue_vel+blue_action
        self.blue_pos,self.blue_vel=CW_Prop(self.blue_pos,self.blue_vel,self.omega,3600)
        self.new_red_pos,self.new_red_vel=self.state_generate(self.red_pos,self.red_vel,self.i)
        self.fuel=np.array([self.fuel[0]-np.linalg.norm(blue_action)])
        self.new_target_pos=self.new_red_pos-self.target_relative_pos
        self.new_target_vel=self.new_red_vel-self.target_relative_vel
        # self.time_delay=np.random.normal(loc=self.delay_mu, scale=self.delay_sigma, size=1)
        # self.delay_pos,self.delay_vel=CW_Prop(self.red_pos,self.red_vel,self.omega,-self.time_delay[0])
        new_state=np.concatenate((self.blue_pos/5e4,self.blue_vel,self.new_target_pos/5e4,self.new_target_vel,self.fuel))
        self.new_distance=np.linalg.norm(self.blue_pos-self.new_target_pos)
        self.new_velocity_difference=np.linalg.norm(self.new_target_vel-self.blue_vel)
        self.red_pos=self.new_red_pos
        self.red_vel=self.new_red_vel
        
        if self.fuel[0]<0 or (self.new_distance<self.target_distance and self.new_velocity_difference<self.velocity_tolerence): # and vel_goal
            self.new_done=True
        if not self.new_done:
            #reward=(np.linalg.norm(natural_blue_pos-self.target_position)-np.linalg.norm(self.blue_pos-self.target_position))/1000-self.distance/5e4-np.linalg.norm(blue_action[0:3])-np.linalg.norm(self.blue_vel-self.target_vel)/10 #+obs_buff
            new_reward=-self.new_distance/3e4-np.linalg.norm(blue_action)#-np.linalg.norm(self.blue_vel-self.target_vel)/30
        if self.new_done:
            if self.fuel[0]<0:
                new_reward=0
            # elif (self.distance<self.target_distance and vel_goal) and self.fuel[0]>=0:
            #     self.info=1
            #     reward=200+self.fuel[0]*5
            #     print(self.blue_vel)
            elif (self.new_distance<self.target_distance) and self.fuel[0]>=0:
                new_reward=400+self.fuel[0]*10-10*self.new_velocity_difference
                #print(self.blue_vel)

        return new_state, new_reward, self.new_done, self.info, self.fuel[0], self.new_distance,self.new_velocity_difference

    def plot(self, args, data_x, data_y, data_z=None):
        font = {'family': 'serif',
         'serif': 'Times New Roman',
         'weight': 'normal',
         'size': 15,
         }
        plt.rc('font', **font)
        plt.style.use('seaborn-whitegrid')
        if data_z!=None and args['plot_type']=="3D-1line":
            fig = plt.figure()
            ax = fig.gca(projection='3d') #Axes对象是图形的绘图区域，可以在其上进行各种绘图操作。
                                        #通过gca()函数可以获取当前图形的Axes对象，如果该对象不存在，则会创建一个新的。
                                        #projection='3d' 参数指定了Axes对象的投影方式为3D，即创建一个三维坐标系。
            plt.plot(-100,0,15,'r*') #画一个位于原点的星形
            plt.plot(data_x,data_y,data_z,'r',linewidth=1) #画三维图
            #ax.scatter(data_x,data_y,data_z,'b',s=1)
            ax.set_xlabel('x/km', fontsize=15)
            ax.set_ylabel('y/km', fontsize=15)
            ax.set_zlabel('z/km', fontsize=15)
            ax.set_xlim(np.min(data_x),np.max(data_x))
            ax.set_ylim(np.min(data_y),np.max(data_y))
            ax.set_zlim(np.min(data_z),np.max(data_z))
            # ax.set_xlim(np.min([np.min(data_x),np.min(data_y),np.min(data_z)]),np.max([np.max(data_x),np.max(data_y),np.max(data_z)]))
            # ax.set_ylim(np.min([np.min(data_x),np.min(data_y),np.min(data_z)]),np.max([np.max(data_x),np.max(data_y),np.max(data_z)]))
            # ax.set_zlim(np.min([np.min(data_x),np.min(data_y),np.min(data_z)]),np.max([np.max(data_x),np.max(data_y),np.max(data_z)]))
            plt.tight_layout()# 调整布局使得图像不溢出
            plt.savefig(args['plot_title'], format='svg', bbox_inches='tight')# 'logs/{}epoch-{}steps.png'.format(epoch,steps))
            plt.show()
        elif data_z!=None and args['plot_type']=="2D-2line":
            fig = plt.figure()
            ax = fig.gca() #Axes对象是图形的绘图区域，可以在其上进行各种绘图操作。通过gca()函数可以获取当前图形的Axes对象，如果该对象不存在，则会创建一个新的。
            plt.plot(data_x,data_y,'b',linewidth=0.5)
            plt.plot(data_x,data_z,'g',linewidth=1)
            ax.set_xlabel('x', fontsize=15)
            ax.set_ylabel('y', fontsize=15)
            ax.set_xlim(np.min(data_x),np.max(data_x))
            ax.set_ylim(np.min([np.min(data_y),np.min(data_z)]),np.max([np.max(data_y),np.max(data_z)]))
            plt.tight_layout()# 调整布局使得图像不溢出
            plt.savefig(args['plot_title'], format='svg', bbox_inches='tight')# 'logs/{}epoch-{}steps.png'.format(epoch,steps))
    
    def plotstep(self,blue_action):
        self.blue_vel=self.blue_vel+blue_action[0:3]
        for i in range(20):
            plot_pos,_=CW_Prop(self.blue_pos,self.blue_vel,self.omega,blue_action[3]/20*i)
            self.blue_pos_list.append(plot_pos)
        self.blue_pos,self.blue_vel=CW_Prop(self.blue_pos,self.blue_vel,self.omega,blue_action[3]) 
        self.blue_pos_list.append(self.blue_pos)
        self.red_pos,self.red_vel=CW_Prop(self.red_pos,self.red_vel,self.omega,blue_action[3])
        self.fuel=np.array([self.fuel[0]-np.linalg.norm(blue_action[0:3])])
        state=np.concatenate((self.blue_pos/5e4,self.blue_vel,self.target_position/5e4,self.target_vel,self.fuel))
        #self.distance=np.linalg.norm(self.blue_pos-self.red_pos)

        self.distance=np.linalg.norm(self.blue_pos-self.target_position)

        if self.blue_vel[0]>0 and self.blue_vel[0]<3 and self.blue_vel[2]>0 and self.blue_vel[2]<3:
            vel_goal=1
        else:
            vel_goal=0

        if self.fuel[0]<0 or (self.distance<self.target_distance): # and vel_goal
            self.done=True

        if self.done:
            if self.fuel[0]<0:
                self.info=0 #用来区分越界和到达目标
            # elif (self.distance<self.target_distance and vel_goal) and self.fuel[0]>=0:
            #     self.info=1
            #     reward=200+self.fuel[0]*5
            #     print(self.blue_vel)
            elif (self.distance<self.target_distance) and self.fuel[0]>=0:
                self.info=1
                print(self.blue_vel)
            self.i+=1

        return state, self.done, self.info, self.blue_pos_list