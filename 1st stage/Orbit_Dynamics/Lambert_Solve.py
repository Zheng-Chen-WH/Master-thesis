'''
Lambert_Solve：求解Lambert问题
输入：初位置矢量r1，末位置矢量r2，初速度v10，转移时间Dt
输出：半长轴a，偏心率e，转移初速度v1，转移末速度v2
'''
import math
import numpy as np


def Lambert_Solve(r1,r2,v10,Dt,mu):
    r1 = np.reshape(r1,3)
    r2 = np.reshape(r2, 3)
    v10 = np.reshape(v10, 3)

    r1_norm = np.linalg.norm(r1)
    r2_norm = np.linalg.norm(r2)
    Dthet = np.arccos(np.dot(r1,r2)/np.linalg.norm(r1)/np.linalg.norm(r2))
    if np.dot(r2,np.cross(np.cross(r1,v10),r1)) < 1:
        Dthet = 2*math.pi-Dthet
    
    a = (r1_norm+r2_norm)/2
    s = math.sqrt(pow(r1_norm,2)+pow(r2_norm,2)-2*r1_norm*r2_norm*math.cos(Dthet))
    deltm = 2*math.asin((r1_norm+r2_norm-s)/(r1_norm+r2_norm+s))
    Dtm = pow((r1_norm+r2_norm+s),1.5)/(8*math.sqrt(mu))*(math.pi-np.sign(math.sin(Dthet))*(deltm-math.sin(deltm)))
    ka = np.sign(Dtm-Dt);  kb = np.sign(math.sin(Dthet))
    tol = 1e-10;  error = 10*tol;  count = 0
    while error > tol and count < 1e3:
        aold = a
        alphst = 2*math.asin(math.sqrt((r1_norm+r2_norm+s)/(4*a)))
        betast = 2*math.asin(math.sqrt((r1_norm+r2_norm-s)/(4*a)))
        alph = math.pi+ka*(alphst-math.pi);  beta = kb*betast
        Fa = math.sqrt(mu)*Dt/pow(a,1.5)-((alph-math.sin(alph))-(beta-math.sin(beta)))
        dalphda = -ka*math.sqrt(r1_norm+r2_norm+s)/(2*math.cos(alphst/2)*pow(a,1.5))
        dbetada = -kb*math.sqrt(r1_norm+r2_norm-s)/(2*math.cos(betast/2)*pow(a,1.5))
        Fpa = -1.5*math.sqrt(mu)*Dt/pow(a,2.5)-(1-math.cos(alph))*dalphda+(1-math.cos(beta))*dbetada
        a = a-Fa/Fpa
        if (r1_norm+r2_norm+s)/(4*a) > 1:
            a = (aold+1/4*(r1_norm+r2_norm+s))/2
        error = abs(a-aold)/a
        count = count+1
    
    p = a*(-r1_norm+r2_norm+s)*(r1_norm-r2_norm+s)/pow(s,2)*pow((math.sin((alph+beta)/2)),2)
    e = math.sqrt(1-p/a)
    
    g = (alph-beta)/2
    cos_f = 1/math.cos(g)*(1-(r1_norm+r2_norm)/2/a)/e
    sin_f = 1/math.sin(g)*(r2_norm-r1_norm)/2/a/e
    f = math.atan2(sin_f,cos_f)
    E1 = f-g
    E2 = f+g
    d_E = E2-E1
    
    # if math.cos(E1/2) == 0:
    #     theta1 = math.pi
    # else:   
    #     theta1 = 2*math.atan(math.sqrt((1+e)/(1-e))*math.tan(E1/2));
    # if math.cos(E2/2) == 0:
    #     theta2 = math.pi
    # else:
    #     theta2 = 2*math.atan(math.sqrt((1+e)/(1-e))*math.tan(E2/2));
    
    v1_r = (math.sqrt(a*mu)*e*math.sin(E1))/(a*(1-e*math.cos(E1)))*r1/r1_norm
    if np.linalg.norm(r2-r1) == 0:
        u1_u = v10
    else:
        u1_u = np.cross(np.cross(r1,r2-r1),r1)
    v1_u = math.sqrt(a*mu*(1-pow(e,2)))/(a*(1-e*math.cos(E1)))*u1_u/np.linalg.norm(u1_u)
    if np.dot(v10,v1_u) < 0:
        v1_u = -v1_u
    v1 = v1_r+v1_u
    
    f1 = -math.sqrt(mu*a)*math.sin(d_E)/r1_norm/r2_norm
    g1 = 1-a/r2_norm*(1-math.cos(d_E))
    v2 = f1*r1+g1*v1

    v1 = np.reshape(v1,(3,1))
    v2 = np.reshape(v2,(3,1))

    return a,e,v1,v2
    
    