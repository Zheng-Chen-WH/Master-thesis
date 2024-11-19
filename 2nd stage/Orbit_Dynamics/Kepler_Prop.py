'''
Kepler_Prop：Kepler轨道预报
输入：半长轴a  偏心率e  轨道倾角i，升交点赤经Omega，近地点幅角omega，真近点角theta:(rad)
输出：位置r  速度v
'''

import math
import numpy as np
from OrbitElements2RV import OrbitElements2RV


def Kepler_Prop(a,e,i,Omega,omega,t_pn,mu):
    r = np.zeros([3,len(t_pn)])
    v = np.zeros([3,len(t_pn)])
    for ii in range(len(t_pn)):
        M = math.sqrt(mu/a**3)*t_pn[ii]
        err = 1
        E1 = M
        while err > 1e-10:
            E2 = E1-(E1-e*math.sin(E1)-M)/(1-e*math.cos(E1))
            err = np.linalg.norm(E2-E1)
            E1 = E2
        E = E2
        if math.cos(E/2) == 0:
            theta = math.pi
        else:
            theta = 2*math.atan(math.sqrt((1+e)/(1-e))*math.tan(E/2))
        if theta < 0:
            theta = theta+2*math.pi
        (ri,vi) = OrbitElements2RV(a,e,i,Omega,omega,theta,mu)
        r[:,ii] = ri.T
        v[:,ii] = vi.T

    return r,v