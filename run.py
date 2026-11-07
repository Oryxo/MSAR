import argparse
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from exp.exp_msar_forecasting import Exp_Msar_Forecast


def parse_args():
    parser = argparse.ArgumentParser(description="Train PatchTST baseline or PatchTST + MSAR.")
    parser.add_argument("--model", default="PatchTST", choices=[
        "PatchTST", "DLinear", "TimeMixer", "iTransformer", "TimesNet",
        "SimpleTM", "DUET",
    ])
    parser.add_argument("--data", default="ETTh1", choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "custom"])
    parser.add_argument("--root_path", default="./dataset/ETT-small")
    parser.add_argument("--data_path", default="ETTh1.csv")
    parser.add_argument("--features", default="M", choices=["M", "S", "MS"])
    parser.add_argument("--target", default="OT")
    parser.add_argument("--freq", default="h")
    parser.add_argument("--seq_len", type=int, default=336)
    parser.add_argument("--label_len", type=int, default=0)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--interval_list", type=lambda s: [int(v) for v in s.split(",")], default=[1])
    parser.add_argument("--seq_len_list", type=lambda s: [int(v) for v in s.split(",")], default=[336])
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--activation", default="gelu")
    parser.add_argument("--embed", default="timeF", choices=["timeF", "fixed", "learned"])
    parser.add_argument("--loss", default="MAE", choices=["MSE", "MAE"])
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lradj", default="type1", choices=["type1", "constant"])
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--checkpoints", default="./checkpoints")
    parser.add_argument("--result_file", default="./results.txt")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--slim_imputation", action="store_true", help="Slim MSAR imputation models only.")
    parser.add_argument("--slim_d_model", type=int, default=16)
    parser.add_argument("--slim_d_ff", type=int, default=32)
    # Shared TimesLib model options.
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--num_kernels", type=int, default=6)
    parser.add_argument("--channel_independence", type=int, default=1)
    parser.add_argument("--down_sampling_layers", type=int, default=0)
    parser.add_argument("--down_sampling_window", type=int, default=1)
    parser.add_argument("--down_sampling_method", default=None, choices=[None, "avg", "max", "conv"])
    parser.add_argument("--decomp_method", default="moving_avg")
    parser.add_argument("--use_norm", type=int, default=1)
    parser.add_argument("--output_attention", action="store_true")
    # SimpleTM options.
    parser.add_argument("--requires_grad", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wv", default="db1")
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--kernel_size", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--l1_weight", type=float, default=5e-5)
    parser.add_argument("--geomattn_dropout", type=float, default=0.5)
    # DUET options. DUET imputation slimming is available but not recommended.
    parser.add_argument("--num_experts", type=int, default=3)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--fc_dropout", type=float, default=0.1)
    parser.add_argument("--CI", dest="CI", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--noisy_gating", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    cli = parse_args()
    if len(cli.interval_list) != len(cli.seq_len_list):
        raise ValueError("interval_list and seq_len_list must have the same number of entries")
    if cli.interval_list[-1] != 1:
        raise ValueError("The final MSAR scale must use interval 1")
    if cli.slim_imputation and len(cli.interval_list) < 2:
        raise ValueError("--slim_imputation applies only when MSAR has imputation scales")

    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    if torch.cuda.is_available() and not cli.no_gpu:
        torch.cuda.manual_seed_all(cli.seed)

    args = SimpleNamespace(**vars(cli))
    args.model = cli.model
    args.model_id = f"{cli.data}_pred{cli.pred_len}"
    args.task_name = "msar_forecasting"
    args.is_training = 1
    csv_path = f"{cli.root_path.rstrip('/')}/{cli.data_path}"
    n_features = len(pd.read_csv(csv_path, nrows=1).columns) - 1
    if cli.features == "S":
        n_features = 1
    args.enc_in = args.dec_in = args.c_out = n_features
    args.use_gpu = not cli.no_gpu
    args.use_multi_gpu = False
    args.use_amp = False
    args.use_dec_inp = False
    args.inverse = False
    args.device = torch.device(f"cuda:{cli.gpu}" if args.use_gpu and torch.cuda.is_available() else "cpu")
    args.gpu_type = "cuda"
    args.itr = 1
    args.des = "patchtst_msar"

    setting = f"{cli.model}_{cli.data}_pred{cli.pred_len}_interval{'-'.join(map(str, cli.interval_list))}_seed{cli.seed}"
    exp = Exp_Msar_Forecast(args)
    print(f"Training setting: {setting}")
    if cli.slim_imputation:
        print(f"Slim imputation enabled: d_model={cli.slim_d_model}, d_ff={cli.slim_d_ff}")
        if cli.model == "DUET":
            print("WARNING: DUET imputation slimming is not recommended; remove --slim_imputation for full-width DUET.")
    exp.train(setting)
    import os
    legacy_result = "result_long_term_forecast.txt"
    previous_size = os.path.getsize(legacy_result) if os.path.exists(legacy_result) else 0
    exp.test(setting)
    if os.path.exists(legacy_result):
        with open(legacy_result, "rb") as source:
            source.seek(previous_size)
            new_result = source.read().decode("utf-8")
        with open(cli.result_file, "a", encoding="utf-8") as destination:
            destination.write(new_result)


if __name__ == "__main__":
    main()
