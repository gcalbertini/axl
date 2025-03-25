import os
import sys
import argparse
import gzip
from utils import *
from train_model import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # GPU train options
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--gpu_num", type=int, default=0)

    # model setting
    parser.add_argument("--llm", type=str, default="opt", help="flan_t5, opt, vicuna")
    parser.add_argument("--recsys", type=str, default="sasrec")

    # train phase setting
    parser.add_argument("--pretrain_stage1", action="store_true")
    parser.add_argument("--pretrain_stage2", action="store_true")
    parser.add_argument("--inference", action="store_true", default=True)
    parser.add_argument("--email_extension", type=str, default=["cvshealth.com"])

    # hyperparameters options
    parser.add_argument("--batch_size1", default=64, type=int)  # match phase 1
    parser.add_argument("--batch_size2", default=2, type=int)
    parser.add_argument("--batch_size_infer", default=2, type=int)
    parser.add_argument("--maxlen", default=57, type=int)
    parser.add_argument("--num_epochs", default=2, type=int)
    parser.add_argument("--stage1_lr", type=float, default=0.0002)
    parser.add_argument("--stage2_lr", type=float, default=0.0002)

    # New argument: combined feature dimension; cardinality of extra features
    parser.add_argument(
        "--combined_feat_dim",
        type=int,
        default=8,
        help="Combined dimension of extra item features (one-hot vectors + additional numeric features); look at previous script for now",
    )

    args = parser.parse_args()

    args.device = "cuda:" + str(args.gpu_num)

    if args.pretrain_stage1:
        train_model_phase1(args)
    elif args.pretrain_stage2:
        train_model_phase2(args)
    elif args.inference:
        inference(args)
