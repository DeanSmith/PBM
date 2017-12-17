# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 10:29:59 2017

@author: Astremon Lu

积分矩方法--the quadrature mothod of moments
"""

from __future__ import division
import numpy as np
from  scipy.integrate import quad, odeint
import matplotlib.pyplot as plt


def PD_algorithm(Moments):
    #矩转换算法----乘积差分算法-the product-difference algorithm
    u = Moments
    # Construct P
    P = np.zeros((7,7))
    P[0, 0] = 1.
    for i in range(1,7):
        P[i-1, 1] = ((-1.)**(i-1)) * u[i-1]
    for j in range(3,8):
        for i in range(1,7):
            P[i-1, j-1] = P[0, j-2]*P[i, j-3] - P[0, j-3]*P[i,j-2]

    # Construct alpha
    alpha = np.zeros(6)
    alpha[0] = 0.
    alpha[1] = u[1]
    for n in range(3, 7):
        alpha[n-1] = P[0, n] / (P[0, n-1] * P[0, n-2])

    # Construct a, b
    a = np.zeros(3)
    b = np.zeros(2)
    for n in range(1, 4):
        a[n-1] = alpha[2*n-1] + alpha[2*n-2]
        if n < 3:
            b[n-1] = np.sqrt(np.abs(alpha[2*n]*alpha[2*n-1]))

    # Construct Jacobian
    J = np.zeros((3, 3))
    J[0, 0] = a[0]
    J[1, 1] = a[1]
    J[2, 2] = a[2]
    J[1, 0] = b[0]
    J[0, 1] = b[0]
    J[1, 2] = b[1]
    J[2, 1] = b[1]

    # Find Absicassas and Wieghts
    r, evectors = np.linalg.eig(J)
    w = np.zeros(3)
    for i in range(3):
        w[i] = u[0] * evectors[0,i]**2
    
    if r[0] < r[1]:
        r = r[::-1]
        w = w[::-1]
    
    return r, w


def pbm_equas_of_moments(y, t, m, r_i, w_i):
    #对通用方程进行矩变换
    nn = 3
    temp_sum1 = np.zeros([nn,nn])
    temp_sum2 = np.zeros([nn,nn])
    for i2 in range(nn):                    #i
        for i3 in range(nn):                #j
            temp_sum1[i2,i3] = w_i[i2]*w_i[i3]*(r_i[i2]**3 + r_i[i3]**3)**(m/3.)  ##
            temp_sum2[i2,i3] = w_i[i2]*w_i[i3]*r_i[i2]**m
    dy =  np.sum(temp_sum1) * 0.5 - np.sum(temp_sum2)
    return dy 


#设置时间序列
compute_time = 1.5
time_steps = 1e-7
time_array = np.arange(0.0, compute_time, time_steps)
time_steps_numbers = len(time_array)
Moments_all = np.zeros([time_steps_numbers,6])
Abscissas = np.zeros([time_steps_numbers,3])
Weights = np.zeros([time_steps_numbers,3])

#初始分布矩计算
#1、需要给出初始分布函数,积分区间（尺寸区间）
for i in range(6):
    integrated_function = lambda r : r**i*(3.*r**2*np.exp(-r**3))
    Moments_all[0,i] = quad(integrated_function, 0., np.inf)[0]

#2、计算各时间节点的矩演化    
for ii in range(time_steps_numbers-1):
    Abscissas[ii,:], Weights[ii,:] = PD_algorithm(Moments_all[ii,:])
    for n in range(6):
        Moments_all[ii+1,n] = odeint(pbm_equas_of_moments,\
                   Moments_all[ii,n],\
                   [time_array[ii],time_array[ii+1]],\
                  args=(n, Abscissas[ii,:], Weights[ii,:]))[-1]

plt.figure(figsize=(16,9))

plt.plot(time_array, Moments_all[:,0]/Moments_all[0,0],label="$M0$")
plt.plot(time_array, Moments_all[:,1]/Moments_all[0,1],label="$M1$")
plt.plot(time_array, Moments_all[:,2]/Moments_all[0,2],label="$M2$")
plt.plot(time_array, Moments_all[:,3]/Moments_all[0,3],label="$M3$")
plt.plot(time_array, Moments_all[:,4]/Moments_all[0,4],label="$M4$")
plt.plot(time_array, Moments_all[:,5]/Moments_all[0,5],label="$M5$")
plt.xlabel('time/s')
plt.ylabel('M_k/M_k_0')
plt.legend()


                                                                                 














































