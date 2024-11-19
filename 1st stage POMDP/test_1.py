import math
import numpy as np
from Orbit_Dynamics.CW_Prop import CW_Prop
from scipy.optimize import fsolve
import time
Omega=2*math.pi/(24*3600)
t=3600
mat_A=np.array([[1,0,6*(Omega*t-math.sin(Omega*t))],
            [0,math.cos(Omega*t),0],
            [0,0,4-3*math.cos(Omega*t)]])
mat_B=np.array([[(4/Omega*math.sin(Omega*t)-3*t),0,2/Omega*(1-math.cos(Omega*t))],
            [0,1/Omega*math.sin(Omega*t),0],
            [2/Omega*(-1+math.cos(Omega*t)),0,1/Omega*math.sin(Omega*t)]])
Xt=np.array([[9.21928081e+03], [1.54413897e+03], [1.00607204e+00]])
Vt=np.array([[ 0.8643822],  [-0.03052425], [-1.51502838]])
Xc=np.array([[-447756.77204467],    [4989.89208884],  [-13417.59216461]])
Vc=np.array([[-1.7221522],   [0.],          [0.01824907]])
uc=np.array([[0],[0],[0]])
# delta_r=A@Xt+B@Vt+B@ut-A@Xc-B@Vc-B@uc
def data():
        delta_v=0.1
        max_X=0
        for alpha in range(int(2*math.pi/0.01)):
            for beta in range(int(2*math.pi/0.01)):
                u1=np.array([[delta_v*math.cos(alpha)*math.cos(beta)],[delta_v*math.sin(beta)],[delta_v*math.sin(alpha)*math.cos(beta)]])
                distance=np.linalg.norm(mat_A@Xt+mat_B@Vt+mat_B@u1-mat_A@Xc-mat_B@Vc-mat_B@uc)
                if distance>max_X:
                        u1_optimal=u1
                        max_X=distance
        return u1_optimal, max_X
# start_1=time.time()
# u1_optimal_A,distance_optimal_A=data()
# end_1=time.time()
# 已知参数
mat_A = mat_A  # 矩阵 A，形状 (m, n)
mat_B = mat_B  # 可逆方阵，形状 (n, n)
gamma = 0.3   # 范数限制 γ,脉冲大小限制

# 计算向量 C
mat_C=mat_A@Xt+mat_B@Vt-mat_A@Xc-mat_B@Vc-mat_B@uc

# 定义目标函数，用于求解 ||u1|| - gamma = 0
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
    return np.linalg.norm(u1)-gamma
# 使用 fsolve 寻找合适的 lambda
lambda_initial_guess = 1

# fsolve 需要目标函数接近0
#fsolve是一种基于牛顿法的根寻找算法，适用于求解一般的非线性方程，总之就是求一个lambda使objective=0，也即|u1|=gamma
lambda_solution, = fsolve(objective, lambda_initial_guess) 

# 计算最终的 u1
matrix_opt = mat_B.T @ mat_B - lambda_solution * np.eye(mat_B.shape[1])
u1_optimal = -np.linalg.solve(matrix_opt, -mat_B.T@mat_C)
end_2=time.time()

# 显示结果
ut=(Xt-Xc).T/np.linalg.norm(Xt-Xc)*gamma
ut=np.array(ut.T)
uv=gamma/np.linalg.norm(Vc)*Vc
print(uv)

print('O1算法认为最优的 u1 为:',u1_optimal.T)
print('位置反向',ut.T)
print(np.linalg.norm(mat_A@Xt+mat_B@Vt+mat_B@u1_optimal-mat_A@Xc-mat_B@Vc-mat_B@uc))
print(np.linalg.norm((mat_B@ut)-mat_B@u1_optimal))
print(math.degrees(math.acos(np.linalg.norm(ut.T*u1_optimal.T)/(np.linalg.norm(u1_optimal)*np.linalg.norm(ut)))))
print(math.degrees(math.acos(np.linalg.norm(uv.T*u1_optimal.T)/(np.linalg.norm(u1_optimal)*np.linalg.norm(uv)))))
# print('MC算法认为最优的u1：',u1_optimal_A.T)
# print(distance_optimal_A)
