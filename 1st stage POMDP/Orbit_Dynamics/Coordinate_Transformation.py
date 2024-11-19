'''
Coordinate_Transformation：坐标转换
变量说明：惯性系下参考航天器位置速度r1,v1，惯性系下相对运动航天器位置速度r2,v2，轨道系下相对运动航天器位置速度dr,dv
'''

import math
import numpy as np


def Inertial2Orbital(r1,v1,r2,v2,mu):
    (Loi,a) = CTM(r1,v1,mu)
    Omega = np.array([0,-math.sqrt(mu/a**3),0])
    dr = Loi@(r2-r1)
    dv = Loi@(v2-v1)-np.cross(Omega,dr.T).T
    return dr,dv


def Orbital2Inertial(r1,v1,dr,dv,mu):
    (Loi,a) = CTM(r1,v1,mu)
    Omega = np.array([0,-math.sqrt(mu/a**3),0])
    r2 = r1+Loi.T@dr
    v2 = v1+Loi.T@dv+Loi.T@np.cross(Omega,dr.T).T
    return r2,v2


def CTM(r1,v1,mu):
    (a,e,i,Omega,omega,theta,E,_,_) = RV2OrbitElements(r1,v1,mu)

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
    Lo1i = np.array([[q11,q12,q13],[q21,q22,q23],[q31,q32,q33]])
    Loo1 = np.array([[0,1,0],[0,0,-1],[-1,0,0]])
    Loi = Loo1@Lo1i

    return Loi,a

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

