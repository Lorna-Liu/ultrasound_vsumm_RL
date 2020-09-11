from PIL import Image
import os
import os.path as osp
import h5py
import cv2
from matplotlib import pyplot as plt
import pylab
import numpy as np
from knapsack import knapsack_dp


def visualize_summary_h5(args,result_file):
#TODO: to be checked if workable
    h5_res = h5py.File(args.savepath + 'results/' + result_file, 'r')
    keys = h5_res.keys()

    for key in keys:
        machine_summary = h5_res[key]['machine_summary'][...]
        fig = plt.figure(figsize=(15, 2))
        ax = fig.add_subplot(111)
        # subplot 0 machine_summary
        N = len(machine_summary)
        ax.plot(range(N), machine_summary, color='red')
        ax.set_xlim(0, N)
        ax.set_yticks([0, 1])
        ax.set_yticklabels([0, 1])
        ax.set_xticklabels([])
        ax.set_xticks([i for i in range(0, N, 2000)])
        ax.set_xticklabels([i for i in range(0, N, 2000)])
        ax.set_title("gt_score")
        pylab.show()
        plt.close()
        # plt.savefig(osp.join(osp.dirname(args.savedir), 'gt_score' + '.png'), bbox_inches='tight')


def visualize_gt_and_summary_h5(args, result_file, gt_key_file):

    gt_key_stem = np.loadtxt(gt_key_file)
    # h5_res = h5py.File(args.savepath + 'results/' + result_file, 'r')
    h5_res = h5py.File(result_file, 'r')
    keys = h5_res.keys()

    for key in keys:
        print(h5_res[key].keys())

        gtscore = h5_res[key]['gtscore'][...]
        Nf = len(gt_key_stem)
        fig, axs = plt.subplots(3, figsize=(12, 3))
        # subplot 0 gt_score together with keyframe index
        axs[0].plot(range(len(gt_key_stem)), 1.1 * gt_key_stem, color='green')
        axs[0].plot(range(len(gtscore)), gtscore, color='red')
        axs[0].set_xlim(0, Nf)
        axs[0].set_yticks([0, 1])
        axs[0].set_yticklabels([0, 1])
        axs[0].set_xticklabels([])
        axs[0].set_xticks([i for i in range(0, Nf, 2000)])
        axs[0].set_xticklabels([i for i in range(0, Nf, 2000)])
        axs[0].set_title("gt_score")


        # subplot 1 ==> User summary score
        machine_summary = h5_res[key]['machine_summary'][...]
        cps = h5_res[key]['cps'][...]
        # transfer cps Idx into stem to plot
        cps = cps[:, 1]

        Ns = max(len(machine_summary), cps[-1])
        if len(machine_summary) < cps[-1]:
            machine_summary2 = np.zeros(Ns)
            machine_summary2[0:len(machine_summary)] = machine_summary
            machine_summary2[len(machine_summary):] = machine_summary2[len(machine_summary)-1]
            machine_summary = machine_summary2
        cps_stem = np.zeros(Ns)
        for cp in range(len(cps)-1):
            cps_stem[cps[cp]] = 1

        axs[1].plot(range(Ns), 1.1 * cps_stem, 'y--')
        axs[1].plot(range(Ns), machine_summary, color='red')
        axs[1].set_xlim(0, Ns)
        axs[1].set_yticks([0, 1])
        axs[1].set_yticklabels([0, 1])
        axs[1].set_xticklabels([])
        axs[1].set_xticks([i for i in range(0, Ns, 2000)])
        axs[1].set_xticklabels([i for i in range(0, Ns, 2000)])

        axs[1].set_title("machine_summary")

        # subplot 2 ==> importance score
        if 'comb_score' in h5_res[key].keys():
            comb_score = h5_res[key]['comb_score'][...]
            axs[2].plot(range(len(comb_score)), comb_score, color='blue')
            axs[2].set_xlim(0, len(comb_score))
        else:
            score = h5_res[key]['score'][...]
            axs[2].plot(range(len(score)), score, color='blue')
            axs[2].set_xlim(0, len(score))
        # axs[2].set_yticks([0, 0.5, 1])
        # axs[2].set_yticklabels([0, 0.5, 1])
        # axs[2].set_xticks([i for i in range(0, n, 2000)])
        # axs[2].set_xticklabels([i for i in range(0, n, 2000)])
        axs[2].set_xticklabels([])
        axs[2].set_title("importance scores")
        pylab.show()
        plt.close()
        fig.savefig(osp.join(osp.dirname(args.savepath), 'figures/demo_1176.png'),bbox_inches=None)
        print ("Done video {}. # frames {}.".format(key, len(machine_summary)))

def visualize_gtscore_txt(args, gt_file):
    gtscore = np.loadtxt(gt_file)
    fig = plt.figure(figsize=(15, 2))
    ax = fig.add_subplot(111)
    # subplot 0 machine_summary
    N = len(gtscore)
    ax.plot(range(N), gtscore, color='red')
    ax.set_xlim(0, N)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([0, 1])
    ax.set_xticklabels([])
    ax.set_xticks([i for i in range(0, N, 2000)])
    ax.set_xticklabels([i for i in range(0, N, 2000)])
    ax.set_title("gt_score")
    pylab.show()
    plt.close()
    # plt.savefig(osp.join(osp.dirname(args.savedir), 'gt_score' + '.png'), bbox_inches='tight')

"""
visualize gt_score and keyframe Idx altogether
"""
def visualize_gt_score_keyIdx_txt(gt_score_file, gt_key_file):
    gt_key_stem = np.loadtxt(gt_key_file)
    gtscore = np.loadtxt(gt_score_file)
    plot_together = True
    N = len(gtscore)

    # stemy = np.ones(len(gt_key_idx))
    # axs[1].stem(gt_key_idx, stemy, color='green')
    if plot_together:
        fig = plt.figure(figsize=(15, 2))
        ax = fig.add_subplot(111)
        ax.plot(range(N), gtscore, color='red')
        ax.set_xlim(0, N)
        ax.set_yticks([0, 1])
        ax.set_yticklabels([0, 1])
        ax.set_xticklabels([])
        ax.set_xticks([i for i in range(0, N, 2000)])
        ax.set_xticklabels([i for i in range(0, N, 2000)])
        ax.set_title("gt_score")
        #
        ax.plot(range(N), 1.1*gt_key_stem, color='green')
    else:
        fig, axs = plt.subplots(2, figsize=(15, 2))
        axs[0].plot(range(N), gtscore, color='red')
        axs[0].set_yticks([0, 1])
        axs[0].set_yticklabels([0, 1])
        axs[0].set_xticklabels([])
        axs[0].set_xticks([i for i in range(0, N, 2000)])
        axs[0].set_xticklabels([i for i in range(0, N, 2000)])
        axs[0].set_title("gt_score")

        axs[1].plot(range(N), 1.1*gt_key_stem, color='green')
        axs[1].set_yticks([0, 1])
        axs[1].set_yticklabels([0, 1])
        axs[1].set_xticklabels([])
        axs[1].set_xticks([i for i in range(0, N, 2000)])
        axs[1].set_xticklabels([i for i in range(0, N, 2000)])
        axs[1].set_title("gt_keyframe")

    pylab.show()
    plt.close()


def visualize_gt_keyframe_txt(args, gt_key_file):
    gt_key_Idx = np.loadtxt(gt_key_file).astype(int)
    nkey = len(gt_key_Idx) # number of keyframes to show
    #np.square(10)
    #hsub = 10
    #vsub = nkey/10+1
    nkey= min(nkey, 36)
    col = int(np.sqrt(nkey))
    row = nkey/col if nkey%col == 0 else (nkey/col + 1)
    fig, axes = plt.subplots(row, col, sharex=True, sharey=True)

    for idx in range(nkey):
        frm_name = str(gt_key_Idx[idx]).zfill(6) + '.jpg'
        frm_path = osp.join(args.frm_dir, frm_name)
        frm = cv2.imread(frm_path)
        frm = cv2.resize(frm, (0, 0), fx=0.1, fy=0.1)
        ax = axes[int(idx / col), int(idx % col)]
        ax.imshow(frm)
        ax.axis("off")
    plt.subplots_adjust(left=0.0, bottom=0.0, right=0.1, top=0.1, wspace=-0.2, hspace=-0.2)
    plt.show()

# this is a backup version
def visualize_gt_keyframe_txt2(args, gt_key_file):
    gt_key_Idx = np.loadtxt(gt_key_file).astype(int)
    nkey = len(gt_key_Idx) # number of keyframes to show
    #np.square(10)
    #hsub = 10
    #vsub = nkey/10+1
    col, row = 4, 4
    fig, axes = plt.subplots(row, col, sharex=True, sharey=True)
    # fig, axs = plt.subplots(hsub, vsub)
    nkey = 16
    # fig, axs = plt.subplots(nkey, figsize=(1, nkey))
    #keyframes =[]

    #ax = fig.add_subplot(111)
    for idx in range(nkey):
        frm_name = str(gt_key_Idx[idx]).zfill(6) + '.jpg'
        frm_path = osp.join(args.frm_dir, frm_name)
        frm = cv2.imread(frm_path)
        frm = cv2.resize(frm, (0, 0), fx=0.1, fy=0.1)
        # keyframes.append(frm)
        #img_array = np.array(frm)
        # for i in range(hsub):
        #     for j in range(vsub):
        #         axs[i, j].imshow(frm)
        #         plt.subplots_adjust(wspace=0, hspace=0)
        ax = axes[int(idx / col), int(idx % col)]
        #axs[idx].imshow(frm)
        #axs[idx].axis("off")
        ax.imshow(frm)
        ax.axis("off")
    plt.subplots_adjust(left=0.0, bottom=0.0, right=0.1, top=0.1, wspace=-0.1, hspace=-0.1)
    plt.show()


def visualize_gt_keyIdx_txt(gt_key_file):
    gt_key_stem = np.loadtxt(gt_key_file)
    # TODO: this function can not work yet
    fig = plt.figure(figsize=(15, 2))
    ax = fig.add_subplot(111)

    N = len(gt_key_stem)
    # stemy = np.ones(len(gt_key_idx))
    # ax.stem(gt_key_idx, stemy, color='red')
    ax.plot(range(N), 1.1 * gt_key_stem, color='green')
    ax.set_xlim(0, N)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([0, 1])
    ax.set_xticklabels([])
    ax.set_xticks([i for i in range(0, N, 2000)])
    ax.set_xticklabels([i for i in range(0, N, 2000)])
    ax.set_title("gt_keyframe")
    pylab.show()
    plt.close()