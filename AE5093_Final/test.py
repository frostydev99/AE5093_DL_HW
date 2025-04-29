from generateTrainingData import generateTrainingData

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

X_train = generateTrainingData(5)
print(X_train)