import torch
import numpy as np
import os
from utils.reranking import re_ranking


def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking

    def reset(self):
        self.feats0 = []
        self.feats1 = []
        self.feats2 = []
        self.feats3 = []
        self.pids = []
        self.camids = []

    def update(self, output):  # called once for each batch
        feat0, feat1, feat2, feat3, pid, camid = output
        self.feats0.append(feat0.cpu())
        self.feats1.append(feat1.cpu())
        self.feats2.append(feat2.cpu())
        self.feats3.append(feat3.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):  # called after each epoch
        feats0 = torch.cat(self.feats0, dim=0)   # torch.Size([2958, 1280])  torch.Size([2962, 1280])
        feats1 = torch.cat(self.feats1, dim=0)   # torch.Size([2958, 1280])  torch.Size([2962, 1280])
        feats2 = torch.cat(self.feats2, dim=0)   # torch.Size([2958, 1280])  torch.Size([2962, 1280])
        feats3 = torch.cat(self.feats3, dim=0)   # torch.Size([2958, 1280])  torch.Size([2962, 1280])
        if self.feat_norm:
            print("The test feature is normalized")
            feats0 = torch.nn.functional.normalize(feats0, dim=1, p=2)  # along channel
            feats1 = torch.nn.functional.normalize(feats1, dim=1, p=2)  # along channel
            feats2 = torch.nn.functional.normalize(feats2, dim=1, p=2)  # along channel
            feats3 = torch.nn.functional.normalize(feats3, dim=1, p=2)  # along channel

        cmc0, mAP0, cmc00, mAP00 = self.cmc_cal(feats0)
        cmc1, mAP1, cmc11, mAP11 = self.cmc_cal(feats1)
        cmc2, mAP2, cmc22, mAP22 = self.cmc_cal(feats2)
        cmc3, mAP3, cmc33, mAP33 = self.cmc_cal(feats3)

        return cmc0, mAP0, cmc00, mAP00, cmc1, mAP1, cmc11, mAP11, cmc2, mAP2, cmc22, mAP22, cmc3, mAP3, cmc33, mAP33

    def cmc_cal(self, feats):
        qf = feats[:self.num_query]  # torch.Size([536, 1280])
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])

        print('=> Computing DistMat with euclidean_distance')
        distmat1 = euclidean_distance(qf, gf)
        distmat2 = euclidean_distance(gf, qf)
        cmc0, mAP0 = eval_func(distmat1, q_pids, g_pids, q_camids, g_camids)
        cmc1, mAP1 = eval_func(distmat2, g_pids, q_pids, g_camids, q_camids)

        return cmc0, mAP0, cmc1, mAP1


