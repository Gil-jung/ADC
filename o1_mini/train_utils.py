import torch
import torch.nn as nn
import torch.optim as optim

def get_loss_criterion():
    return nn.CrossEntropyLoss()

def get_optimizer(model, learning_rate=0.001):
    return optim.Adam(model.parameters(), lr=learning_rate)

def get_scheduler(optimizer, step_size=7, gamma=0.1):
    return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)