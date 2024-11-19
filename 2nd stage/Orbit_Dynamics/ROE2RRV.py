'''
ROE2RRV：相对轨道要素转速度位置矢量
输入：参考航天器半长轴a0，漂移率D(rad/s)，相对偏心率，相对倾角，平纬度辐角差，时间
输出：相对位置，相对速度
'''

import math
import numpy as np


def ROE2RRV(a0,D,dex,dey,dix,diy,dM,t,mu):
    Omega = math.sqrt(mu/a0**3)
    du = Omega*t
    x = a0*dM+2*a0*(dex*math.sin(du)-dey*math.cos(du))+a0*D*t
    y = a0*(dix*math.cos(du)+diy*math.sin(du))
    z = 2*a0*D/3/Omega+a0*(dex*math.cos(du)+dey*math.sin(du))
    vx = a0*D+2*a0*Omega*(dex*math.cos(du)+dey*math.sin(du))
    vy = a0*Omega*(diy*math.cos(du)-dix*math.sin(du))
    vz = a0*Omega*(-dex*math.sin(du)+dey*math.cos(du))

    dr = np.array([[x],[y],[z]])
    dv = np.array([[vx], [vy], [vz]])

    return dr,dv

