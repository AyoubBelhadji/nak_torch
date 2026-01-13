#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# This file contains the implementation of the Himmelblau's potential
# Ayoub Belhadji
# 05/12/2025


import torch




def himmelblau(T):
    def himmelblau_aux(x):
        x1, x2 = x[:,0], x[:,1]
        
        t1 = torch.pow(torch.pow(x1,2)+x2-11,2)
        t2 = torch.pow(x1 + torch.pow(x2,2)-7,2)
        
        Ux = t1 + t2
        return -Ux/T
    return himmelblau_aux




