'''
轨道要素转位置速度矢量
输入：半长轴a:(m)  偏心率e  轨道倾角i，升交点赤经Omega，近地点幅角omega，真近点角theta:(rad)
输出：位置r:(m)  速度v:(m/s)
'''

import math
import numpy as np


def OrbitElements2RV(a,e,i,Omega,omega,theta):
    mu=398600441800000.0
    r_norm = a*(1-pow(e,2))/(1+e*math.cos(theta))
    p = a*(1-pow(e,2))
    u = omega+theta
    q11 = math.cos(u)*math.cos(Omega)-math.sin(u)*math.cos(i)*math.sin(Omega)
    q12 = math.cos(u)*math.sin(Omega)+math.sin(u)*math.cos(i)*math.cos(Omega)
    q13 = math.sin(u)*math.sin(i)
    q21 = -math.sin(u)*math.cos(Omega)-math.cos(u)*math.cos(i)*math.sin(Omega)
    q22 = -math.sin(u)*math.sin(Omega)+math.cos(u)*math.cos(i)*math.cos(Omega)
    q23 = math.cos(u)*math.sin(i)
    q31 = math.sin(i)*math.sin(Omega)
    q32 = -math.sin(i)*math.cos(Omega)
    q33 = math.cos(i)
    Loi = np.array([[q11,q12,q13],[q21,q22,q23],[q31,q32,q33]])
    Lio = np.transpose(Loi)
    r = Lio@np.array([[r_norm],[0],[0]])
    v = math.sqrt(mu/p)*np.array([[q11*e*math.sin(theta)+q21*(1+e*math.cos(theta))],[q12*e*math.sin(theta)+q22*(1+e*math.cos(theta))],[q13*e*math.sin(theta)+q23*(1+e*math.cos(theta))]])
    r=r.T.flatten()
    v=v.T.flatten()
    return r,v

