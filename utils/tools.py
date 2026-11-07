import numpy as np
import torch


def adjust_learning_rate(optimizer, epoch, args, scheduler=None, printout=True):
    if args.lradj == "constant":
        lr = args.learning_rate
    else:
        lr = args.learning_rate * (0.5 ** (epoch - 1))
    for group in optimizer.param_groups:
        group["lr"] = lr
    if printout:
        print(f"Updating learning rate to {lr}")


class EarlyStopping:
    def __init__(self, patience=3, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None or score >= self.best_score + self.delta:
            self.best_score = score
            self.counter = 0
            self.val_loss_min = val_loss
            models = model if isinstance(model, list) else [model]
            paths = path if isinstance(path, list) else [path]
            for item, item_path in zip(models, paths):
                torch.save(item.state_dict(), f"{item_path}/checkpoint.pth")
            if self.verbose:
                print(f"Validation loss improved; checkpoint saved ({val_loss:.6f})")
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
