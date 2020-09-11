from __future__ import print_function
import torch
import argparse
import sys, os
import os.path as osp
import h5py
from train_utils import save_model_epoch
from models import DSN
from train_eval import train
from train_utils import Logger, read_json, weights_init
from evaluate import evaluate


parser = argparse.ArgumentParser("Pytorch code for ultrasound video summarization using reinforcement learning")
parser.add_argument('-s', '--split', type=str, required=False, help="path to split file",
                            default="../datasets/us_dataset/splits_50.json")
parser.add_argument('--split-id', type=int, default=0, help="split index (default: 0)")
parser.add_argument('-g', '--gtpath', type=str, required=False, help="path to txt gtscores",
                                default="../datasets/us_dataset/gt_scores/")
parser.add_argument('--train-model', type=str, default='unsup', choices=['sup', 'unsup'], help="(training model)")
parser.add_argument('--reward-type', type=str, default='Rall', choices=['Rdet', 'RrepRdet', 'RdivRdet', 'Rall'],
                                                                help="Reward type (default: Rdiv)")
parser.add_argument('--comb-score', action='store_false', help="whether to combine sononet detection scores")
parser.add_argument('--lr', type=float, default=1e-4, help="learning rate (default: 1e-05)")
parser.add_argument('--proportion', type=float, default=0.15, help="proportion(default: 0.15)")
parser.add_argument('--hidden-dim', type=int, default=256, help="hidden unit dimension of DSN (default: 256)")
parser.add_argument('--num-layers', type=int, default=1, help="number of RNN layers (default: 1)")
parser.add_argument('--rnn-cell', type=str, default='lstm', help="RNN cell type (default: lstm)")
# Optimization options
parser.add_argument('--weight-decay', type=float, default=1e-05, help="weight decay rate (default: 1e-05)")
parser.add_argument('--max-epoch', type=int, default=300, help="maximum epoch for training (default: 60)")
parser.add_argument('--stepsize', type=int, default=60, help="how many steps to decay learning rate (default: 30)")
parser.add_argument('--gamma', type=float, default=0.5, help="learning rate decay (default: 0.5)")
parser.add_argument('--num-episode', type=int, default=5, help="number of episodes (default: 5)")
parser.add_argument('--beta', type=float, default=0.1, help="weight for summary length penalty term (default: 0.01)")
# Misc
parser.add_argument('--seed', type=int, default=1, help="random seed (default: 1)")
parser.add_argument('--gpu', type=str, default='0', help="which gpu devices to use")
parser.add_argument('--save-dir', type=str, default='../output/', help="path to save output (default: 'log')")
parser.add_argument('--resume', type=str, default='', help="path to resume file")
parser.add_argument('--verbose', action='store_true', help="whether to show detailed test results")


args = parser.parse_args()

torch.manual_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_gpu = torch.cuda.is_available()


if __name__ == '__main__':

    BASE_DATA_PATH = '../datasets/us_dataset/h5_files/'
    dataset = {}

    # load the video data ( features saved in .h5 format)
    print("Loading all dataset: ")
    for subj in os.listdir(BASE_DATA_PATH):
        h5data_path = os.path.join(BASE_DATA_PATH, subj)
        if not os.path.exists(h5data_path):
            print("The dataset for subj %s doesn't exist. Skipping..." % subj)
            continue

        dataset[subj[:10]] = h5py.File(h5data_path, 'r')

    splits = read_json(args.split)
    split = splits[args.split_id]

    num_train_vids = len(split['train_keys'])
    num_test_vids = len(split['test_keys'])
    specif_path = 'train_' + str(args.split_id) + '_' + args.reward_type
    args.save_path = osp.join(args.save_dir, args.train_model, 'split' + str(args.split_id))
    sys.stdout = Logger(osp.join(args.save_path, 'log_' + args.reward_type + str(args.lr)
                                 + args.train_model + '.txt'))

    model = DSN(in_dim=64, hid_dim=args.hidden_dim, num_layers=args.num_layers, cell=args.rnn_cell)
    model.apply(weights_init)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    print(" ========== \nArgs:{} \n========== ".format(args))
    args.train_keys = split['train_keys']
    args.test_keys = split['test_keys']

    if args.train_model == 'sup':
        print("========Supervised Learning========")
    else:
        args.use_reward = True
        print("========Unsupervised Learning========")

    args.demo_h5 = osp.join(args.save_path, 'h5_res' + args.reward_type + str(args.lr) +
                            args.train_model)

    model = train(args, model, dataset)

    # Testing
    Fscore, Precision, Recall = evaluate(args, model, dataset)

    # save model
    save_model_epoch(args, model, args.max_epoch)


