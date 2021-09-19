#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 09:53:54 2021

@author: cghiaus

Simple wall with capacities in all temperature nodes
Inputs: outdoor temperature, indoor convection heat flow rate (from HVAC)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dm4bem

# Physical properties
# ===================
wall = {'Conductivity': [1.400, 0.040],
        'Density': [2300.0, 16.0],
        'Specific heat': [880, 1210],
        'Width': [0.2, 0.08],
        'Slices': [4, 2]}
wall = pd.DataFrame(wall, index=['Concrete', 'Insulation'])

air = {'Density': 1.2,
       'Specific heat': 1000}

# convection coefficients, W/m² K
h = pd.DataFrame([{'in': 4., 'out': 10}])

S_wall = 3 * 3      # m²
V_air = 3 * 3 * 3   # m³

# Thermal resistances:
# conduction
R_cd = wall['Width'] / (wall['Conductivity'] * S_wall)
# convection
R_cv = 1 / (h * S_wall)

# Thermal capacities:
C_wall = wall['Density'] * wall['Specific heat'] * wall['Width'] * S_wall
C_air = air['Density'] * air['Specific heat'] * V_air

# Differential algebraic equations (DAE)
# ======================================
# number of temperature nodes and flow branches
no_t = no_q = sum(wall['Slices']) + 1

# Conductance matrix
R = np.zeros([no_q])
R[0] = R_cv['out'] + R_cd['Concrete'] / 8
R[1] = R[2] = R[3] = R_cd['Concrete'] / 4
R[4] = R_cd['Concrete'] / 8 + R_cd['Insulation'] / 4
R[5] = R_cd['Insulation'] / 2
R[6] = R_cd['Insulation'] / 4 + R_cv['in']
G = np.diag(np.reciprocal(R))

# Capacity matrix
C = np.zeros(no_t)
# C[0] = C[1] = C[2] = C[3] = 1 / wall['Slices']['Concrete'] * S_wall \
#     * wall['Width']['Concrete'] \
#     * wall['Density']['Concrete'] \
#     * wall['Specific heat']['Concrete']
# C[4] = C[5] = 1 / wall['Slices']['Insulation'] * S_wall \
#     * wall['Width']['Insulation'] \
#     * wall['Density']['Insulation'] \
#     * wall['Specific heat']['Insulation']
# C[6] = V_air * air['Density'] * air['Specific heat']
# C1 = np.diag(C)

C[0] = C[1] = C[2] = C[3] = C_wall['Concrete'] / 4
C[4] = C[5] = C_wall['Insulation'] / 2
C[6] = C_air
C = np.diag(C)


# Arc-node incidence matrix
# A = np.eye(no_q + 1, no_t)
# A = -np.transpose(np.diff(A, n=1, axis=0))
A = np.eye(no_q, no_t + 1)
A = -np.diff(A, n=1, axis=1)

b = np.zeros(no_q)
f = np.zeros(no_t)

# Steady state solution
# =====================
# for To = 1 °C
b[0] = 1
temp_steady_To = np.linalg.inv(A.T @ G @ A) @ (A.T @ G @ b + f)

# for heating flux = 1 W
b[0] = 0
f[-1] = 1
temp_steady_Qh = np.linalg.inv(A.T @ G @ A) @ (A.T @ G @ b + f)

# State-space model
# =================
# u = [b; f]    input vector
B = np.linalg.inv(C) @ np.block([A.T @ G, np.eye(no_t)])
# Select columns for which the input vector is not zero
# 1st for To and last for Qh
B = B[:, [0, -1]]

# State matrix
A = np.linalg.inv(C) @ (-A.T @ G @ A)

# Output matrix
C = np.zeros((1, no_t))
# output: last temperature node
C[:, -1] = 1

# Feedthrough (or feedforward) matrix
D = np.zeros(B.shape[1])

# Eigenvalues, stability and time step
# ====================================
max_dt = min(-2 / np.linalg.eig(A)[0])
print(f'Max time step dt: {max_dt:.2f} s')
dt = 360
dt = 416.5

# Time integration using Euler
# ============================
filename = 'FRA_Lyon.074810_IWEC.epw'
start_date = '2000-04-10'
end_date = '2000-05-15'

# Read weather data from Energyplus .epw file
[data, meta] = dm4bem.read_epw(filename, coerce_year=None)
weather = data[["temp_air", "dir_n_rad", "dif_h_rad"]]
del data
weather.index = weather.index.map(lambda t: t.replace(year=2000))
weather = weather[(weather.index >= start_date) & (
    weather.index < end_date)]

days = weather.shape[0] / 24

# number of steps
n = int(np.floor(3600 / dt * 24 * days))
n = int(np.ceil(3600 / dt * 24 * days))
# time
t = np.arange(0, n * dt, dt)

fig, axs = plt.subplots(2, 2)

# Step input: outdoor temperature To
# ----------------------------------
u = np.block([[np.ones([1, n])],
              [np.zeros([1, n])]])

# initial values for temperatures obtained by explicit and implicit Euler
temp_exp = np.zeros([no_t, t.shape[0]])
temp_imp = np.zeros([no_t, t.shape[0]])
for k in range(t.shape[0] - 1):
    temp_exp[:, k + 1] = (np.eye(no_t) + dt * A) @\
        temp_exp[:, k] + dt * B @ u[:, k]
    temp_imp[:, k + 1] = np.linalg.inv(np.eye(no_t) - dt * A) @\
        (temp_imp[:, k] + dt * B @ u[:, k])

# axs[0, 0].plot(t / 3600, temp_exp[-1, :], t / 3600, temp_imp[-1, :])
axs[0, 0].plot(t / 3600, temp_exp[-1, :], t / 3600, temp_imp[-1, :])
# axs[0, 0].set_ylabel('Air temperature [°C]')
# axs[0, 0].set_title('Step input: To')
axs[0, 0].set(ylabel='Air temperature [°C]', title='Step input: To')

# Step input: heat flow Qh
# ------------------------
u = np.block([[np.zeros([1, n])],
              [np.ones([1, n])]])

# initial values for temperatures obtained by explicit and implicit Euler
temp_exp = np.zeros([no_t, t.shape[0]])
temp_imp = np.zeros([no_t, t.shape[0]])
# for k in range(n - 1):
for k in range(t.shape[0] - 1):
    temp_exp[:, k + 1] = (np.eye(no_t) + dt * A) @\
        temp_exp[:, k] + dt * B @ u[:, k]
    temp_imp[:, k + 1] = np.linalg.inv(np.eye(no_t) - dt * A) @\
        (temp_imp[:, k] + dt * B @ u[:, k])

axs[0, 1].plot(t / 3600, temp_exp[-1, :], t / 3600, temp_imp[-1, :])
axs[0, 1].set_title('Step input: Qh')

# Simulation with outdoor temperature
# -----------------------------------

# time for weather (in seconds)
tw = np.arange(0, 3600 * weather.shape[0], 3600)
# change the timestep from 1h to dt
t = np.arange(0, 3600 * weather.shape[0], dt)
# outdoor temperature at timestep dt
temp_out = np.interp(t, tw, weather['temp_air'])

# integration by Euler explicit and tmplicit

u = np.block([[temp_out],
             [np.zeros(temp_out.shape[0])]])

# initial values for temperatures obtained by explicit and implicit Euler
temp_exp = np.zeros([no_t, n])
temp_imp = np.zeros([no_t, n])
temp_exp = np.zeros([no_t, t.shape[0]])
temp_imp = np.zeros([no_t, t.shape[0]])
for k in range(n - 1):
    temp_exp[:, k + 1] = (np.eye(no_t) + dt * A) @\
        temp_exp[:, k] + dt * B @ u[:, k]
    temp_imp[:, k + 1] = np.linalg.inv(np.eye(no_t) - dt * A) @\
        (temp_imp[:, k] + dt * B @ u[:, k])

axs[1, 0].plot(t / 3600, temp_exp[-1, :],
               t / 3600, temp_out)
axs[1, 0].set_xlabel('Time [hours]')
axs[1, 0].set_ylabel('Air temperature [°C]')
axs[1, 0].set_title('Explicit Euler')

axs[1, 1].plot(t / 3600, temp_imp[-1, :],
               t / 3600, temp_out)
axs[1, 1].set_xlabel('Time [hours]')
axs[1, 1].set_ylabel('Air temperature [°C]')
axs[1, 1].set_title('Implicit Euler')

fig.tight_layout()
