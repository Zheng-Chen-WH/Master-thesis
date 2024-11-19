from Orbit_Dynamics.CW_Prop import CW_Prop
import math #角度转弧度;反余弦;弧度转角度
import numpy as np
from scipy.optimize import minimize
import time as run_time

class FirstStage():
    def __init__(self,seed=None):
        self.target_position=np.array([-100000,0,15000])
        self.target_distance=5e3
        self.omega=2*math.pi/(24*3600)
        self.GM=3.986e14
        self.radius=(self.GM/self.omega**2)**(1/3)
        self.i=0
        self.info=0
        self.impulse_num=10
        self.seed=seed

    def reset(self): #,prop_t
        if self.seed is None:
            np.random.seed(self.i)
        else:
            np.random.seed(self.seed)
        super().__init__()
        self.blue_pos_list=[]
        #self.red_pos_list=[]
        self.info=0
        self.target_vel=np.array([np.random.uniform(1, 3),
                                 np.random.uniform(0, 0),
                                 np.random.uniform(1, 3)])
        self.blue_arc=np.random.uniform(-0.008285772983018781,-0.010653136692452719)
        self.blue_radius=self.radius+np.random.uniform(-5e4,0)
        self.blue_pos=np.array([self.blue_radius*math.sin(self.blue_arc),
                                np.random.uniform(-5,5)*1000,
                                self.radius-self.blue_radius*math.cos(self.blue_arc)])
        self.blue_abs_vel=(self.GM/self.blue_radius)**(1/2)
        self.ref_vel=(self.GM/self.radius)**(1/2)
        self.vel_noise=np.random.uniform(-0.05,0.05)
        self.blue_vel=np.array([self.blue_abs_vel*math.cos(self.blue_arc)+self.omega*self.blue_pos[2]-self.ref_vel, #+self.vel_noise
                                 0.0,#np.random.uniform(-0.0, 0.0),
                                 self.blue_abs_vel*math.sin(self.blue_arc)-self.omega*self.blue_pos[0]]) #+self.vel_noise
        self.red_arc=np.random.uniform(-0.00023673637094339374,0.00023673637094339374)
        self.red_pos=np.array([self.radius*math.sin(self.red_arc),
                                np.random.uniform(-2,2)*1000,
                                self.radius-self.radius*math.cos(self.red_arc)])
        self.red_vel=np.array([self.ref_vel*math.cos(self.red_arc)-self.ref_vel+self.omega*self.red_pos[2]+self.vel_noise,
                                 np.random.uniform(-0.05, 0.05),
                                 self.ref_vel*math.sin(self.red_arc)-self.omega*self.red_pos[2]+self.vel_noise])
        self.blue_pos_list.append(self.blue_pos)
        # self.red_pos_list.append(self.red_pos)
        self.done=False
        self.fuel=np.array([50])
        self.init_distance=np.linalg.norm((self.blue_pos-self.target_position))
        self.distance=self.init_distance
        state=np.concatenate((self.blue_pos,self.blue_vel,self.target_position,self.fuel))
        self.passed=False
        return state

    def Main(self):
        dv=self.Optimal(self.impulse_num)
        dis,fuel=self.propagator(dv,self.impulse_num)
        return dis,fuel,dv

    def propagator(self,dv,N):
        blue_pos=self.blue_pos
        blue_vel=self.blue_vel
        fuel=0 
        for i in range(N):
            # 开始预报带脉冲轨道
            fuel+=np.linalg.norm(dv[N+3*i:N+3*i+3])
            blue_vel=blue_vel+dv[N+3*i:N+3*i+3]
            blue_pos,blue_vel=CW_Prop(blue_pos,blue_vel,self.omega,dv[i])
        end_distance=np.linalg.norm(blue_pos-self.target_position)
        return end_distance,fuel

    def restraint(self,x):
        distance_target=1000
        n=int(len(x)/4)
        dv_sequence=x
        distance_achieved,_=self.propagator(dv_sequence,n)
        distance_error=distance_target-distance_achieved #不等式约束只看返回值是否大于等于0，大于等于0说明满足约束，所以需要改写约束不等式
        return [distance_error]

    def TargetFunction(self,x):
        n = int(len(x) / 4)
        dv_used = 0
        dv_sequence = x[n:]
        for i in range(n):
            dv_used=dv_used+np.linalg.norm(dv_sequence[i * 3:i * 3 + 3])
        return dv_used

    def Optimal(self,N):
        n = N
        time_x0=np.ones(n)*2000
        amp_x0=np.zeros(3*n)
        #amp_x0=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,1.9,1.8,1.7,1.6,1.5,1.4,1.3,1.2,1.1,1.0])
        x0=np.concatenate((time_x0,amp_x0))
        #生成一整个大数组，前面n-1个是喷气间隔，后面2n个是每次喷气的某方向大小，n次x,y喷气,用作优化起始点；整个数组也是变量；xyz方向都有
        time_lower_boundary=np.ones(n)*600
        time_upper_boundary=np.ones(n)*3000
        amp_lower_boundary = -np.ones(3*n)*2 #时间与推力下界
        amp_upper_boundary = np.ones(3*n)*2 #上界
        lower_boundary=np.concatenate((time_lower_boundary,amp_lower_boundary))
        upper_boundary=np.concatenate((time_upper_boundary,amp_upper_boundary))
        bounds = [(lower, upper) for lower, upper in zip(lower_boundary, upper_boundary)]
        result = minimize(self.TargetFunction, x0, method='SLSQP',tol=0.1, bounds=bounds, constraints=({'type': 'ineq', 'fun': self.restraint}), options={'maxiter': 5000,'disp':False})
        #maxiter:优化器最大迭代次数
        #优化器返回结果包括：The optimization result represented as a ``OptimizeResult`` object.
        # Important attributes are: ``x`` the solution array, ``success`` a Boolean flag indicating if the optimizer exited successfully and
        #``message`` which describes the cause of the termination.
        # # #print("收敛情况：",result.success)
        opt_dv_sequence=result.x
        return opt_dv_sequence
start=run_time.time()
avg_fuel=0
for i in range(1000):
    env=FirstStage(i)
    env.reset()
    dis,fuel,dv=env.Main()
    avg_fuel+=fuel
end=run_time.time()
print((end-start)/1000)
print(avg_fuel/1000)
    