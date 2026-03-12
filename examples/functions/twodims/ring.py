#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# This file contains a typical 2D mixture of ring
# covariance matrices
# Ayoub Belhadji
# 05/12/2025


import torch



def ring(T, r=2.0, sigma=0.5):
    def ring_aux(x):
        x1, x2 = x[..., 0], x[..., 1]
        Ux = torch.square(torch.sqrt(torch.square(x1) + torch.square(x2)) - r) / (2 * sigma**2)
        return -Ux / T
    return ring_aux