#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 08:56:15 2022

@author: cghiaus

Comparrison methods: MNA  with U_bat for steady state
"""
import numpy as np

To = -5     # °C outdoor temperature
Tisp = 18   # °C (desired = set point) indoor temperature

λ = 1.4     # W/(m K) thermal conductivity
w = 0.15    # m width
S = 3       # m^2 surface area
U = λ / w   # W/(m^2 K) overall heat transfer coefficient

# U_bat method
# Inverse problem
q0 = U * S * (To - Tisp)
q1 = -q0
print(f"q_Ubat = {q1:.2f} W")


# MCA method
# inverse problem as a control problem
def direct_problem(Kp):
    Kp = 10 ** float(exp)
    A = np.array([[1],
                  [1]])
    G = np.array([[U * S, 0],
                  [0, Kp]])
    b = np.array([To, Tisp])
    f = 0

    θ = np.linalg.inv(A.T @ G @ A) @ (A.T @ G @ b + f)
    q = G @ (-A @ θ + b)
    return θ, q


print(f"{'Kp': >9} {'θ [°C]': >9} {'q [W]': >9}")
for exp in np.arange(-4, 8):
    Kp = 10 ** float(exp)
    θ, q = direct_problem(Kp)
    print(f"{Kp: >9.0e} {θ[0]: >9.2f} {q[1]: >9.2f}")
