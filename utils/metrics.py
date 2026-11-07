import numpy as np


def metric(pred, true):
    mae = np.mean(np.abs(pred - true))
    mse = np.mean((pred - true) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((pred - true) / true))
    mspe = np.mean(((pred - true) / true) ** 2)
    return mae, mse, rmse, mape, mspe
