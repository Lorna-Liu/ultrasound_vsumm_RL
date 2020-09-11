from __future__ import print_function
import torch
import os.path as osp
import time
import datetime
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.distributions import Bernoulli
from train_utils import save_model_epoch
from rewards import compute_reward_det_coff
import numpy as np
from evaluate import evaluate


def train(args, model, dataset):

    lr = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint)
    else:
        start_epoch = 0

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Currently using GPU {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
        model = nn.DataParallel(model).cuda()
    else:
        print("Currently using CPU")

    print("=====> Start training <===== ")
    start_time = time.time()
    model.train()

    train_keys = args.train_keys
    baselines = {key: 0. for key in train_keys} # baseline rewards for videos
    reward_writers = {key: [] for key in train_keys} # record reward changes for each video

    for epoch in range(start_epoch, args.max_epoch):
        idxs = np.arange(len(train_keys))
        np.random.shuffle(idxs)  # shuffle indices

        for idx in idxs:
            key = train_keys[idx]
            video_info = dataset[key][u'video_1']
            seq = video_info['features'][...] # sequence of features, (seq_len, dim)
            seq = torch.from_numpy(seq).unsqueeze(0) # input shape (1, seq_len, dim)
            if use_gpu:
                seq = seq.cuda()

            sig_probs = model(seq)

            reg_loss = args.beta * (sig_probs.mean() - 0.5) ** 2

            m = Bernoulli(sig_probs)
            epis_rewards = []

            if args.train_model == 'sup':
                # loading label for supervised training
                gtscore = np.loadtxt(osp.join(args.gtpath, 'gtscore_' + key + '_5fps.txt'))
                positions = video_info['picks'][...]
                label = gtscore[positions]

                sum_loss = sum_loss_MSE(sig_probs, label)

            else: # unsupervised mode
                 sum_loss = 0

            cost = reg_loss + sum_loss

            for episode in range(args.num_episode):
                actions = m.sample()
                log_probs = m.log_prob(actions)

                if args.reward_type == 'Rdet' or args.reward_type == 'Rall' or args.reward_type == 'RrepRdet'\
                        or args.reward_type == 'RdivRdet':

                    det_scores = dataset[key][u'video_1']['det_scores'][...]
                    det_class = dataset[key][u'video_1']['det_class'][...]
                    if args.reward_type == 'Rdet':  # include the detection reward
                        div_coff= 0.0
                        rep_coff = 0.0
                        det_coff = 1.0
                    elif args.reward_type == 'RrepRdet':
                        div_coff = 0.0
                        rep_coff= 1.0
                        det_coff = 1.0
                    elif args.reward_type == 'RdivRdet':
                        div_coff= 1.0
                        det_coff = 1.0
                        rep_coff = 0.0
                    elif args.reward_type == 'Rall':
                        rep_coff= 1.0
                        det_coff= 1.0
                        div_coff = 1.0

                    reward = compute_reward_det_coff(seq, actions, det_scores, det_class, episode,
                                use_gpu=use_gpu, div_coff=div_coff, rep_coff=rep_coff, det_coff=det_coff)

                    expected_reward = log_probs.mean() * (reward - baselines[key])
                    cost = cost - 10 * expected_reward  # minimize negative expected reward
                    epis_rewards.append(reward.item())

            baselines[key] = 0.9 * baselines[key] + 0.1 * np.mean(epis_rewards) # update baseline reward via moving average
            reward_writers[key].append(np.mean(epis_rewards))

            optimizer.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        epoch_reward = np.mean([reward_writers[key][epoch] for key in train_keys])

        if (epoch + 1) % 5 == 0:
            if args.train_model == 'sup':
                print("epoch {}/{}\t reward {}\t  sum_loss {}\t reg_loss{} \t cost {}\t".format(epoch + 1,
                                                 args.max_epoch,  epoch_reward,  sum_loss, reg_loss, cost))
            else:
                print("epoch {}/{}\t reward {}\t reg_loss {} \t  cost {}\t".format(epoch + 1,
                                                  args.max_epoch, epoch_reward, reg_loss, cost))

        if (epoch + 1) % 50 == 0:
            Fscore, Precision, Recall = evaluate(args, model, dataset, args.demo_h5, epoch)

            print("epoch:{:0>3d}\t F45:{:.2%}\t P45:{:.2%}\t R45:{:.2%}\t ".format(epoch + 1,
                                                              Fscore[3], Precision[3], Recall[3]))

        # save_model_epoch(args, model, epoch, Fscore[0]*100)
        if epoch + 1 == args.max_epoch or epoch + 1 == 50 or epoch + 1 == 100:
            save_model_epoch(args, model, epoch, Fscore[0]*100)
        model.train()
        scheduler.step()

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))
    return model

def sum_loss_MSE(pred_score, gt_labels):
    gt_labels = gt_labels.reshape(-1)
    gt_labels = torch.tensor(gt_labels, dtype=torch.float32)
    gt_labels = gt_labels.cuda()

    if pred_score.dim() > 1:
        pred_score = pred_score.squeeze(0).squeeze(1)
    criterion = nn.MSELoss()
    loss = criterion(pred_score, gt_labels)

    return loss

