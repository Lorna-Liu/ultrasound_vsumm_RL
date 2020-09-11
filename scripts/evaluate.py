import torch
import h5py
import os.path as osp
import vsum_tools, eval_tools
from tabulate import tabulate
import numpy as np


def evaluate(args, model, dataset, proportion=0.15):

    with torch.no_grad():
        model.eval()
        meanFscores = []
        meanPrecision = []
        meanRecall = []

        if args.verbose: table = [["No.", "Video", "F-score"]]
        sum_proportions = [0.15, 0.25, 0.35, 0.45]
        for key_idx, key in enumerate(args.test_keys):

            # load gt_score txt file
            gtscore = np.loadtxt(osp.join(args.gtpath, 'gtscore_' + key + '_5fps.txt'))

            video_info = dataset[key][u'video_1']
            positions = video_info['picks'][...]

            seq = video_info['features'][...]
            seq = torch.from_numpy(seq).unsqueeze(0)

            seq = seq.cuda()
            sig_probs = model(seq)

            probs_importance = sig_probs.data.cpu().squeeze().numpy()

            # generate gt summary on-fly according to different proportion constrains
            fscore_prop = []
            precision_prop = []
            recall_prop = []
            for p in range(len(sum_proportions)):
                prop = sum_proportions[p]
                # generate gt summary on-fly according to different propoption constrains
                gt_summary, _, _ = vsum_tools.generate_summary(gtscore, video_info, positions, proportion=prop)
                machine_summary, pred_probs_up, pick_segs \
                            = vsum_tools.generate_summary(probs_importance, video_info, positions, proportion=prop)
                f_score, precision, recall = eval_tools.evaluate_summary(machine_summary, gt_summary)
                fscore_prop.append(f_score)
                precision_prop.append(precision)
                recall_prop.append(recall)

            meanFscores.append(fscore_prop)
            meanPrecision.append(precision_prop)
            meanRecall.append(recall_prop)

            if args.verbose:
                table.append([key_idx+1, key, "{:.1%}".format(f_score)])

    if args.verbose:
        print(tabulate(table))

    mean_fscores = np.mean(meanFscores, axis=0)
    mean_precision = np.mean(meanPrecision, axis=0)
    mean_recall = np.mean(meanRecall, axis=0)
    return mean_fscores, mean_precision, mean_recall
