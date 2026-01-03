import torch
from torch.utils.data import DataLoader
import numpy as np

import utils.spatial_transforms as ST
import utils.temporal_transforms as TT
import utils.transforms as T
import utils.seqtransforms as SeqT
# from torchvision.transforms import InterpolationMode
# import torchvision.transforms as T

from datasets.video_loader_xh import VideoDataset, VideoDataset_lmdb
from datasets.video_loader_vvi import VideoDataset_train, VideoDataset_train_lmdb
# from datasets.samplers import RandomIdentitySampler, RandomIdentitySamplerForSeq, RandomIdentitySamplerWYQ
from datasets.sampler_vvi import IdentitySampler
from datasets.seqpreprocessor import SeqTrainPreprocessor, SeqTestPreprocessor

from datasets.set.bupt_lmdb import BUPT_lmdb
# from datasets.set.bupt import BUPT
from datasets.set.vcm import VCM
from datasets import IterLoader

__factory = {
    'vcm': VCM,
    'bupt_lmdb': BUPT_lmdb,
    # 'bupt': BUPT
}


def GenIdx(train_color_label, train_thermal_label):
    color_pos = []
    unique_label_color = np.unique(train_color_label)  # 1074 ge xun lian ID
    for i in range(len(unique_label_color)):
        tmp_pos = [k for k, v in enumerate(train_color_label) if v == unique_label_color[i]]
        color_pos.append(tmp_pos)  # 1074 ge list

    thermal_pos = []
    unique_label_thermal = np.unique(train_thermal_label)  # 1074
    for i in range(len(unique_label_thermal)):
        tmp_pos = [k for k, v in enumerate(train_thermal_label) if v == unique_label_thermal[i]]
        thermal_pos.append(tmp_pos)   # 1074 ge list
    return color_pos, thermal_pos


def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    imgs_ir, pid_ir, camid_ir, imgs_rgb, pid_rgb, camid_rgb
    """
    imgs_ir, pid_ir, camid_ir, imgs_rgb, pid_rgb, camid_rgb = zip(*batch)
    irs = torch.stack(imgs_ir, dim=0)
    rgbs = torch.stack(imgs_rgb, dim=0)
    pids_ir = torch.tensor(pid_ir, dtype=torch.int64)
    pids_rgb = torch.tensor(pid_rgb, dtype=torch.int64)
    camids_ir = torch.tensor(camid_ir, dtype=torch.int64)
    camids_rgb = torch.tensor(camid_rgb, dtype=torch.int64)
    return irs, rgbs, pids_ir, pids_rgb, camids_ir, camids_rgb


def train_collate_fn2(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    imgs_ir, pid_ir, camid_ir, imgs_rgb, pid_rgb, camid_rgb
    """
    imgs_rgb, pid_rgb, camid_rgb = zip(*batch)
    rgbs = torch.stack(imgs_rgb, dim=0)
    pids_rgb = torch.tensor(pid_rgb, dtype=torch.int64)

    camids_rgb = torch.tensor(camid_rgb, dtype=torch.int64)
    return rgbs, pids_rgb, camids_rgb


def val_collate_fn(batch):
    imgs, pids, camids = zip(*batch)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch


def train_collate_fn_seq(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, flows, pids, camids = zip(*batch)
    viewids = 1
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids,


def val_collate_fn_seq(batch):
    imgs, flows, pids, camids = zip(*batch)
    viewids = 1
    img_paths = None
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths


def make_dataloader_imgs(cfg):
    seq_len = cfg.INPUT.SEQ_LEN  # 10
    num_workers = cfg.DATALOADER.NUM_WORKERS  # 8

    dataset = __factory[cfg.DATASETS.NAMES]()

    rgb_pos, ir_pos = GenIdx(dataset.rgb_label, dataset.ir_label)
    num_classes = dataset.num_train_pids  # 1074
    cam_num = dataset.num_train_cams  # 6
    view_num = 1  # 1

    transform_train = SeqT.Compose([SeqT.RectScale(cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1]),
                                    SeqT.RandomHorizontalFlip(),
                                    SeqT.RandomSizedEarser(),
                                    SeqT.ToTensor(),
                                    SeqT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    transform_test = SeqT.Compose([SeqT.RectScale(cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1]),
                                   SeqT.ToTensor(),
                                   SeqT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    sampler = IdentitySampler(dataset.ir_label, dataset.rgb_label, rgb_pos, ir_pos, 2, 8)
    index1 = sampler.index1  # ndarray,all tracklets for rgb modality 4288
    index2 = sampler.index2  # ndarray,all tracklets for ir modality 4288

    train_set = VideoDataset_train(dataset.train_ir, dataset.train_rgb, seq_len=seq_len, sample='video_train',
                                   transform=transform_train, index1=index1, index2=index2)
    # train_set_normal = VideoDataset(dataset.train, seq_len=seq_len, sample='dense', transform=transform_test)
    train_set_normal_rgb = VideoDataset(dataset.train_rgb, seq_len=seq_len, sample='rrs_train',
                                        transform=transform_test)
    train_set_normal_ir = VideoDataset(dataset.train_ir, seq_len=seq_len, sample='rrs_train', transform=transform_test)


    # trainloader = DataLoader(
    # 	VideoDataset_train(),
    # 	sampler=sampler,
    # 	batch_size=loader_batch, num_workers=args.workers,
    # 	drop_last=True,
    # )
    train_loader_stage2 = IterLoader(DataLoader(
        train_set,
        sampler=sampler,
        batch_size=8,
        # batch_size=1,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=train_collate_fn,
    ), length=300)

    train_loader_stage0_rgb = DataLoader(
        train_set_normal_rgb,
        # batch_size=cfg.SOLVER.STAGE1.IMS_PER_BATCH,
        batch_size=128,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
        collate_fn=train_collate_fn2
    )
    train_loader_stage0_ir = DataLoader(
        train_set_normal_ir,
        # batch_size=cfg.SOLVER.STAGE1.IMS_PER_BATCH,
        batch_size=128,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
        collate_fn=train_collate_fn2
    )

    sampler_method = 'rrs_test'
    batch_size_eval = 128
    evalloader1 = DataLoader(
        VideoDataset(dataset.query+dataset.gallery, seq_len=seq_len, sample=sampler_method, transform=transform_test),
        batch_size=batch_size_eval, shuffle=False, num_workers=num_workers, drop_last=False, collate_fn=val_collate_fn)


    # evalloader2 = DataLoader(
    #     VideoDataset(dataset.query_1 + dataset.gallery_1, seq_len=seq_len, sample=sampler_method, transform=transform_test),
    #     batch_size=batch_size_eval, shuffle=False, num_workers=num_workers, drop_last=False, collate_fn=val_collate_fn)

    loader_list = [train_loader_stage2,
                         train_loader_stage0_rgb,
                         train_loader_stage0_ir,
                         evalloader1,
                         ]
    return loader_list, len(dataset.query), len(dataset.query_1), num_classes, cam_num


def make_dense_eval_dataloader_imgs_vcm(cfg):
    seq_len = cfg.INPUT.SEQ_LEN  # 10
    num_workers = cfg.DATALOADER.NUM_WORKERS  # 8

    dataset = __factory[cfg.DATASETS.NAMES]()

    rgb_pos, ir_pos = GenIdx(dataset.rgb_label, dataset.ir_label)
    num_classes = dataset.num_train_pids  # 1074
    cam_num = dataset.num_train_cams  # 6
    view_num = 1  # 1

    transform_test = SeqT.Compose([SeqT.RectScale(cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1]),
                                   SeqT.ToTensor(),
                                   SeqT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    sampler_method = 'dense'
    batch_size_eval = 1
    evalloader1 = DataLoader(
        VideoDataset(dataset.query+dataset.gallery, seq_len=seq_len, sample=sampler_method, transform=transform_test),
        batch_size=batch_size_eval, shuffle=False, num_workers=num_workers, drop_last=False, collate_fn=val_collate_fn)

    return evalloader1, len(dataset.query), len(dataset.query_1), num_classes, cam_num


def make_rrs_eval_dataloader_imgs_vcm(cfg):
    seq_len = cfg.INPUT.SEQ_LEN  # 10
    num_workers = cfg.DATALOADER.NUM_WORKERS  # 8

    dataset = __factory[cfg.DATASETS.NAMES]()

    rgb_pos, ir_pos = GenIdx(dataset.rgb_label, dataset.ir_label)
    num_classes = dataset.num_train_pids  # 1074
    cam_num = dataset.num_train_cams  # 6
    view_num = 1  # 1

    transform_test = SeqT.Compose([SeqT.RectScale(cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1]),
                                   SeqT.ToTensor(),
                                   SeqT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    sampler_method = 'rrs_test'
    batch_size_eval = 32
    evalloader1 = DataLoader(
        VideoDataset(dataset.query+dataset.gallery, seq_len=seq_len, sample=sampler_method, transform=transform_test),
        batch_size=batch_size_eval, shuffle=False, num_workers=num_workers, drop_last=False, collate_fn=val_collate_fn)

    return evalloader1, len(dataset.query), len(dataset.query_1), num_classes, cam_num

def make_dataloader_lmdb(cfg):
    seq_len = cfg.INPUT.SEQ_LEN  # 10
    num_workers = cfg.DATALOADER.NUM_WORKERS  # 8

    dataset = __factory[cfg.DATASETS.NAMES]()

    rgb_pos, ir_pos = GenIdx(dataset.rgb_label, dataset.ir_label)
    num_classes = dataset.num_train_pids  # 1074
    cam_num = dataset.num_train_cams  # 6
    view_num = 1  # 1

    transform_train = SeqT.Compose([SeqT.RectScale(cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1]),
                                    SeqT.RandomHorizontalFlip(),
                                    SeqT.RandomSizedEarser(),
                                    SeqT.ToTensor(),
                                    SeqT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    transform_test = SeqT.Compose([SeqT.RectScale(cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1]),
                                   SeqT.ToTensor(),
                                   SeqT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    sampler = IdentitySampler(dataset.ir_label, dataset.rgb_label, rgb_pos, ir_pos, 2, 8)
    index1 = sampler.index1  # ndarray,all tracklets for rgb modality 4288
    index2 = sampler.index2  # ndarray,all tracklets for ir modality 4288

    train_set = VideoDataset_train(dataset.train_ir, dataset.train_rgb, seq_len=seq_len, sample='video_train',
                                   transform=transform_train, index1=index1, index2=index2)
    # train_set_normal = VideoDataset(dataset.train, seq_len=seq_len, sample='dense', transform=transform_test)
    train_set_normal_rgb = VideoDataset_lmdb(dataset.train_rgb, seq_len=seq_len, sample='rrs_train',
                                        transform=transform_test)
    train_set_normal_ir = VideoDataset_lmdb(dataset.train_ir, seq_len=seq_len, sample='rrs_train', transform=transform_test)


    # trainloader = DataLoader(
    # 	VideoDataset_train(),
    # 	sampler=sampler,
    # 	batch_size=loader_batch, num_workers=args.workers,
    # 	drop_last=True,
    # )
    train_loader_stage2 = IterLoader(DataLoader(
        train_set,
        sampler=sampler,
        batch_size=8,
        # batch_size=1,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=train_collate_fn,
    ), length=300)

    train_loader_stage0_rgb = DataLoader(
        train_set_normal_rgb,
        # batch_size=cfg.SOLVER.STAGE1.IMS_PER_BATCH,
        batch_size=30,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
        collate_fn=train_collate_fn2
    )
    train_loader_stage0_ir = DataLoader(
        train_set_normal_ir,
        # batch_size=cfg.SOLVER.STAGE1.IMS_PER_BATCH,
        batch_size=30,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
        collate_fn=train_collate_fn2
    )

    sampler_method = 'rrs_test'
    batch_size_eval = 32
    evalloader1 = DataLoader(
        VideoDataset_lmdb(dataset.query+dataset.gallery, seq_len=seq_len, sample=sampler_method, transform=transform_test),
        batch_size=batch_size_eval, shuffle=False, num_workers=num_workers, drop_last=False, collate_fn=val_collate_fn)


    evalloader2 = DataLoader(
        VideoDataset_lmdb(dataset.query_1 + dataset.gallery_1, seq_len=seq_len, sample=sampler_method, transform=transform_test),
        batch_size=batch_size_eval, shuffle=False, num_workers=num_workers, drop_last=False, collate_fn=val_collate_fn)

    loader_list = [train_loader_stage2,
                         train_loader_stage0_rgb,
                         train_loader_stage0_ir,
                         evalloader1,
                         evalloader2,
                         ]
    return loader_list, len(dataset.query), len(dataset.query_1), num_classes, cam_num, view_num


def make_eval_all_dataloader(cfg):
    split_id = cfg.DATASETS.SPLIT
    seq_srd = cfg.INPUT.SEQ_SRD
    seq_len = cfg.INPUT.SEQ_LEN
    num_workers = cfg.DATALOADER.NUM_WORKERS

    if cfg.DATASETS.NAMES != 'mars' and cfg.DATASETS.NAMES != 'duke' and cfg.DATASETS.NAMES != 'lsvid':

        dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR, split_id=split_id, seq_len=seq_len,
                                                seq_srd=seq_srd, num_val=1)

        num_classes = dataset.num_trainval_ids
        cam_num = dataset.num_train_cams
        view_num = dataset.num_train_vids

        val_set = SeqTestPreprocessor(dataset.query + dataset.gallery, dataset, seq_len,
                                      transform=SeqT.Compose([SeqT.RectScale(256, 128),
                                                              SeqT.ToTensor(),
                                                              SeqT.Normalize(mean=[0.485, 0.456, 0.406],
                                                                             std=[0.229, 0.224, 0.225])]))
        val_loader = DataLoader(
            val_set,
            batch_size=cfg.TEST.IMS_PER_BATCH,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            collate_fn=val_collate_fn_seq
        )

    else:
        dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

        transform_test = SeqT.Compose([SeqT.RectScale(cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1]),
                                       SeqT.ToTensor(),
                                       SeqT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        num_classes = dataset.num_train_pids  # 625
        cam_num = dataset.num_train_cams  # 6
        view_num = dataset.num_train_vids  # 1

        sampler_method = 'dense'
        batch_size_eval = 1

        val_set = VideoDataset(dataset.query + dataset.gallery, seq_len=seq_len, sample=sampler_method,
                               transform=transform_test)

        val_loader = DataLoader(
            val_set,
            batch_size=batch_size_eval,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            drop_last=False,
            collate_fn=val_collate_fn
        )

    return val_loader, len(dataset.query), num_classes, cam_num, view_num


def make_eval_rrs_dataloader_bupt(cfg):
    seq_len = cfg.INPUT.SEQ_LEN
    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES]()

    transform_test = SeqT.Compose([SeqT.RectScale(cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1]),
                                   SeqT.ToTensor(),
                                   SeqT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    num_classes = dataset.num_train_pids  # 625
    cam_num = dataset.num_train_cams  # 6
    view_num = 1  # 1

    sampler_method = 'rrs_test'
    batch_size_eval = 32
    evalloader_rgb2ir = DataLoader(
        VideoDataset(dataset.query + dataset.gallery, seq_len=seq_len, sample=sampler_method, transform=transform_test),
        batch_size=batch_size_eval, shuffle=False, num_workers=num_workers, drop_last=False, collate_fn=val_collate_fn)

    evalloader_ir2rgb = DataLoader(
        VideoDataset(dataset.query_1 + dataset.gallery_1, seq_len=seq_len, sample=sampler_method,
                     transform=transform_test),
        batch_size=batch_size_eval, shuffle=False, num_workers=num_workers, drop_last=False, collate_fn=val_collate_fn)

    return evalloader_rgb2ir, evalloader_ir2rgb, len(dataset.query), len(dataset.query_1), num_classes, cam_num


def make_eval_rrs_dataloader_bupt_lmdb(cfg):
    seq_len = cfg.INPUT.SEQ_LEN
    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES]()

    transform_test = SeqT.Compose([SeqT.RectScale(cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1]),
                                   SeqT.ToTensor(),
                                   SeqT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    num_classes = dataset.num_train_pids  # 625
    cam_num = dataset.num_train_cams  # 6
    view_num = 1  # 1

    sampler_method = 'rrs_test'
    batch_size_eval = 32
    evalloader_rgb2ir = DataLoader(
        VideoDataset_lmdb(dataset.query + dataset.gallery, seq_len=seq_len, sample=sampler_method, transform=transform_test),
        batch_size=batch_size_eval, shuffle=False, num_workers=num_workers, drop_last=False, collate_fn=val_collate_fn)

    evalloader_ir2rgb = DataLoader(
        VideoDataset_lmdb(dataset.query_1 + dataset.gallery_1, seq_len=seq_len, sample=sampler_method,
                     transform=transform_test),
        batch_size=batch_size_eval, shuffle=False, num_workers=num_workers, drop_last=False, collate_fn=val_collate_fn)

    return evalloader_rgb2ir, evalloader_ir2rgb, len(dataset.query), len(dataset.query_1), num_classes, cam_num
