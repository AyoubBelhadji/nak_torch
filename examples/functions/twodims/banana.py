#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# This file contains the implementation of the banana
# Ayoub Belhadji
# 05/12/2025


import torch

def banana(T):
    def banana_aux(x):
        x1, x2 = x[..., 0], x[..., 1]
        Ux = 0.5 * (torch.square(x1) / 4 + torch.square(x2 - torch.square(x1)))
        return -Ux / T
    return banana_aux
