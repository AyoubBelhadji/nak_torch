#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# This file contains the implementation of the KDE in PyTorch
# Ayoub Belhadji
# 05/12/2025


import torch



def gaussian_kde(sigma, dataset):
    data = torch.as_tensor(dataset, dtype=torch.float32)
    if data.ndim == 1: 
        data = data[:, None]         

    N, d = data.shape

    def gaussian_kde_aux(x):
        x = torch.as_tensor(x, dtype=data.dtype, device=data.device)
        if x.ndim == 1:
            x = x[:, None]           
        diff = x[:, None, :] - data[None, :, :]     # (M, N, d)
        sqdist = (diff**2).sum(-1)                  # (M, N)
        return torch.log(torch.exp(- sqdist / (sigma*sigma)).mean(1))

    return gaussian_kde_aux