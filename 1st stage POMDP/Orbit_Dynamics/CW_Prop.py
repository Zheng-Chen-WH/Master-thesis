'''
CW_Prop：CW方程预报
输入：初始位置速度，角速度，时间
输出：末端位置速度
'''

import math
import numpy as np


def CW_Prop(dr0,dv0,Omega,t):
    dr0 = np.reshape(dr0,3)
    dv0 = np.reshape(dv0, 3)
    x0 = dr0[0]
    y0 = dr0[1]
    z0 = dr0[2]
    vx0 = dv0[0]
    vy0 = dv0[1]
    vz0 = dv0[2]

    x = x0+2*vz0/Omega*(1-math.cos(Omega*t))+(4*vx0/Omega-6*z0)*math.sin(Omega*t)+(6*Omega*z0-3*vx0)*t
    y = y0*math.cos(Omega*t)+vy0/Omega*math.sin(Omega*t)
    z = 4*z0-2*vx0/Omega+(2*vx0/Omega-3*z0)*math.cos(Omega*t)+vz0/Omega*math.sin(Omega*t)
    vx = (4*vx0-6*Omega*z0)*math.cos(Omega*t)+2*vz0*math.sin(Omega*t)+6*Omega*z0-3*vx0
    vy = vy0*math.cos(Omega*t)-y0*Omega*math.sin(Omega*t)
    vz = vz0*math.cos(Omega*t)+(3*Omega*z0-2*vx0)*math.sin(Omega*t)

    dr = np.array([x, y, z])
    dv = np.array([vx, vy, vz])

    return dr,dv