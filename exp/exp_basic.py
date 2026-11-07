import torch


class Exp_Basic:
    def __init__(self, args):
        self.args = args

    def _acquire_device(self):
        if self.args.use_gpu and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.args.gpu}")
            print(f"Use GPU: {device}")
            return device
        print("Use CPU")
        return torch.device("cpu")
