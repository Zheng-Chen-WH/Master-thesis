'''
位置速度矢量转轨道要素
输入：位置r:(m)  速度v:(m/s)
输出：半长轴a:(m)  偏心率e  轨道倾角i，升交点赤经Omega，近地点幅角omega，真近点角theta，偏近点角E:(rad)
'''

import math
import numpy as np


def RV2OrbitElements(r,v,mu):
    r = np.reshape(r,3)
    v = np.reshape(v, 3)

    E_t = pow(np.linalg.norm(v),2)/2-mu/np.linalg.norm(r)
    a = -mu/2/E_t
    H = np.cross(r,v)
    p = pow(np.linalg.norm(H),2)/mu
    if p > a:
        e_norm = 0
    else:
        e_norm = math.sqrt(1 - p / a)
    b = np.array([-H[1],H[0],0])
    e = np.cross(v,H)/mu-r/np.linalg.norm(r)
    i = math.acos(H[2]/np.linalg.norm(H))
    Omega = math.atan2(b[1],b[0])
    if Omega < 0:
        Omega = Omega+2*math.pi
    omega = math.acos(np.dot(b,e)/np.linalg.norm(b)/np.linalg.norm(e))
    if e[2] <= 0:
        omega = 2*math.pi-omega
    u = math.acos(np.dot(b,r)/np.linalg.norm(b)/np.linalg.norm(r))
    if r[2] <= 0:
        u = 2*math.pi-u
    theta = u-omega
    if theta < 0:
        theta = theta+2*math.pi
    if math.cos(theta/2) == 0:
        E = math.pi
    else:
        E = 2*math.atan(math.sqrt((1-e_norm)/(1+e_norm))*math.sin(theta/2)/math.cos(theta/2))
    if E < 0:
        E = E+2*math.pi
    M = E-e_norm*math.sin(E)
    t_pn = M/math.sqrt(mu/a**3)

    return a,e_norm,i,Omega,omega,theta,E,M,t_pn

