import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from utils.iotools import save_checkpoint
from torch.cuda import amp
import torch.distributed as dist
from torch.nn import functional as F
from loss.supcontrast import SupConLoss
from loss.softmax_loss import CrossEntropyLabelSmooth
from model.cm import ClusterMemory


def do_train_stage2(cfg,
             model,
             center_criterion,
             loader_list,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query1,
             num_query2,
            local_rank,
                    num_classes):
    log_period = cfg.SOLVER.STAGE2.LOG_PERIOD
    eval_period = cfg.SOLVER.STAGE2.EVAL_PERIOD
    train_loader_stage2, train_loader_stage0_rgb, train_loader_stage0_ir, evalloader_rgb2ir = loader_list

    device = "cuda"
    epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS

    logger = logging.getLogger("TFCLIP.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    loss_meter_cm_rgb = AverageMeter()
    loss_meter_cm_ir = AverageMeter()
    acc_meter = AverageMeter()
    acc_meter_id1 = AverageMeter()
    acc_meter_id2 = AverageMeter()

    evaluator_rgb2ir = R1_mAP_eval(num_query1, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    # evaluator_ir2rgb = R1_mAP_eval(num_query2, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    # scaler = amp.GradScaler()
    xent_frame = nn.CrossEntropyLoss()

    @torch.no_grad()
    def generate_cluster_features(labels, features):
        import collections
        centers = collections.defaultdict(list)
        for i, label in enumerate(labels):
            if label == -1:
                continue
            centers[labels[i]].append(features[i])

        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]

        centers = torch.stack(centers, dim=0)
        return centers

    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()
    #######   1.CLIP-Memory module ####################

    best_performance = 0.0
    best_epoch = 1
    for epoch in range(1, epochs + 1):
        print("=> Automatically generating CLIP-Memory (might take a while, have a coffe)")
        image_features_rgb = []
        image_features_ir = []
        labels_ir = []
        labels_rgb = []
        with torch.no_grad():
            for n_iter, (img, vid, target_cam) in enumerate(train_loader_stage0_rgb):
                img = img.to(device)  # torch.Size([30, 10, 3, 288, 144])
                target = vid.to(device)  # torch.Size([64])
                if len(img.size()) == 6:
                    # method = 'dense'
                    b, n, s, c, h, w = img.size()
                    assert (b == 1)
                    img = img.view(b * n, s, c, h, w)  # torch.Size([5, 8, 3, 256, 128])
                    # with amp.autocast(enabled=True):
                    image_feature = model(img, get_image=True)
                    image_feature = image_feature.view(-1, image_feature.size(1))
                    image_feature = torch.mean(image_feature, 0, keepdim=True)  # 1,512
                    for i, img_feat in zip(target, image_feature):
                        labels_rgb.append(i)
                        image_features_rgb.append(img_feat.cpu())
                else:
                    # with amp.autocast(enabled=True):
                    image_feature = model(img, get_image=True)
                    for i, img_feat in zip(target, image_feature):
                        labels_rgb.append(i)
                        image_features_rgb.append(img_feat.cpu())
        
            for n_iter, (img, vid, target_cam) in enumerate(train_loader_stage0_ir):
                img = img.to(device)  # torch.Size([30, 10, 3, 288, 144])
                target = vid.to(device)  # torch.Size([64])
                if len(img.size()) == 6:
                    # method = 'dense'
                    b, n, s, c, h, w = img.size()
                    assert (b == 1)
                    img = img.view(b * n, s, c, h, w)  # torch.Size([5, 8, 3, 256, 128])
                    # with amp.autocast(enabled=True):
                    image_feature = model(img, get_image=True)
                    image_feature = image_feature.view(-1, image_feature.size(1))
                    image_feature = torch.mean(image_feature, 0, keepdim=True)  # 1,512
                    for i, img_feat in zip(target, image_feature):
                        labels_ir.append(i)
                        image_features_ir.append(img_feat.cpu())
                else:
                    # with amp.autocast(enabled=True):
                    image_feature = model(img, get_image=True)
                    for i, img_feat in zip(target, image_feature):
                        labels_ir.append(i)
                        image_features_ir.append(img_feat.cpu())
        
            labels_list_rgb = torch.stack(labels_rgb, dim=0).cuda()  # N torch.Size([8256])
            labels_list_ir = torch.stack(labels_ir, dim=0).cuda()  # N torch.Size([8256])
            image_features_list_rgb = torch.stack(image_features_rgb, dim=0).cuda()  # torch.Size([3574, 1280])
            image_features_list_ir = torch.stack(image_features_ir, dim=0).cuda()  # torch.Size([3574, 1280])
        
        cluster_features_rgb = generate_cluster_features(labels_list_rgb.cpu().numpy(), image_features_list_rgb)
        cluster_features_ir = generate_cluster_features(labels_list_ir.cpu().numpy(), image_features_list_ir)
        
        memory_rgb = ClusterMemory(1280, len(set(labels_list_rgb)), temp=0.05,
                               momentum=0.1, use_hard=True).cuda()  # torch.Size([1074, 1280])
        memory_rgb.features = F.normalize(cluster_features_rgb, dim=1).cuda()  # torch.Size([1074, 1280])
        
        memory_ir = ClusterMemory(1280, len(set(labels_list_ir)), temp=0.05,
                                   momentum=0.1, use_hard=True).cuda()  # torch.Size([1074, 1280])
        memory_ir.features = F.normalize(cluster_features_ir, dim=1).cuda()  # torch.Size([1074, 1280])

        start_time = time.time()
        loss_meter.reset()
        loss_meter_cm_rgb.reset()
        loss_meter_cm_ir.reset()
        acc_meter.reset()
        acc_meter_id1.reset()
        acc_meter_id2.reset()
        evaluator_rgb2ir.reset()
        # evaluator_ir2rgb.reset()

        model.train()
        # for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage2):
        for n_iter in range(len(train_loader_stage2)):
            irs, rgbs, pids_ir, pids_rgb, camids_ir, camids_rgb = train_loader_stage2.next()
            img = torch.cat((irs, rgbs), dim=0)  # torch.Size([16, 10, 3, 288, 144])
            vid = torch.cat((pids_ir, pids_rgb))  #
            target_cam = torch.cat((camids_ir, camids_rgb))
            optimizer.zero_grad()
            # optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            if cfg.MODEL.SIE_CAMERA:
                target_cam = target_cam.to(device)
            else:
                target_cam = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else:
                target_view = None
            # with amp.autocast(enabled=True):
            B, T, C, H, W = img.shape  # B=64, T=4.C=3 H=256,W=128
            score, feat, logits1 = model(x = img, cam_label=target_cam)
            score1 = score[:2]
            score2 = score[-2]
            score3 = score[-1]
#             score4 = score[-3]
            loss1 = loss_fn(score1, feat, target, target_cam, isprint=False)

            targetX = target.unsqueeze(1)  # 12,1   => [94 94 10 10 15 15 16 16 75 75 39 39]
            targetX = targetX.expand(B, T)
            # 12,8  => [ [94...94][94...94][10...10][10...10] ... [39...39] [39...39]]
            targetX = targetX.contiguous()
            targetX = targetX.view(B * T,
                                   -1)  # 96  => [94...94 10...10 15...15 16...16 75...75 39...39]
            targetX = targetX.squeeze(1)
            loss_frame1 = xent_frame(score2, targetX)
            loss_frame2 = xent_frame(score3, targetX)
#             loss_frame3 = xent_frame(score4, targetX)

            #
            loss_rgb = memory_rgb(logits1, target)
            loss_ir = memory_ir(logits1, target)

            loss = loss1 + loss_rgb + loss_ir + loss_frame1 / T + loss_frame2 / T


            # scaler.scale(loss).backward()
            loss.backward()
            optimizer.step()
            # scaler.step(optimizer)
            # scaler.update()


            acc1 = (logits1.max(1)[1] == target).float().mean()
            acc_id1 = (score[0].max(1)[1] == target).float().mean()
            acc_id2 = (score[3].max(1)[1] == targetX).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            loss_meter_cm_rgb.update(loss_rgb.item(), img.shape[0])
            loss_meter_cm_ir.update(loss_ir.item(), img.shape[0])
            acc_meter.update(acc1, 1)
            acc_meter_id1.update(acc_id1, 1)
            acc_meter_id2.update(acc_id2, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Loss_rgb: {:.3f}, Loss_ir: {:.3f},Acc_clip: {:.3f}, Acc_id1: {:.3f}, Acc_id2: {:.3f}, Base Lr: {:.2e}"
                    .format(epoch, (n_iter + 1), len(train_loader_stage2),
                            loss_meter.avg,
                            loss_meter_cm_rgb.avg,
                            loss_meter_cm_ir.avg,
                            acc_meter.avg,
                            acc_meter_id1.avg,
                            acc_meter_id2.avg,
                            scheduler.get_lr()[0]))

        scheduler.step()

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, B / time_per_batch))


        if epoch % eval_period == 0:
            model.eval()
            ######### 1.rgb2ir eval
            for n_iter, (img, pid, camid, camids,) in enumerate(evalloader_rgb2ir):
                img = img.to(device)  # torch.Size([64, 4, 3, 256, 128])
                with torch.no_grad():
                    img = img.to(device)
                    if cfg.MODEL.SIE_CAMERA:
                        camids = camids.to(device)

                    out_feat_all, feat0, feat1, feat2 = model(img, cam_label=camids)
                    # feat = feat.view(-1, feat.size(1))
                    # feat = torch.mean(feat, 0, keepdim=True)  # 1,512
                    evaluator_rgb2ir.update((out_feat_all, feat0, feat1, feat2, pid, camid))

            cmc0, mAP0, cmc00, mAP00, cmc1, mAP1, cmc11, mAP11, cmc2, mAP2, cmc22, mAP22, cmc3, mAP3, cmc33, mAP33 = evaluator_rgb2ir.compute()
            logger.info("Validation Results IR2RGB ")
            logger.info("mAP: {:.1%}".format(mAP0))
            for r in [1, 5, 10, 20]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc0[r - 1]))
            logger.info("Validation Results RGB2IR ")
            logger.info("mAP: {:.1%}".format(mAP00))
            for r in [1, 5, 10, 20]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc00[r - 1]))

            logger.info("Validation Results IR2RGB ")
            logger.info("mAP: {:.1%}".format(mAP1))
            for r in [1, 5, 10, 20]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc1[r - 1]))
            logger.info("Validation Results RGB2IR ")
            logger.info("mAP: {:.1%}".format(mAP11))
            for r in [1, 5, 10, 20]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc11[r - 1]))

            logger.info("Validation Results IR2RGB ")
            logger.info("mAP: {:.1%}".format(mAP2))
            for r in [1, 5, 10, 20]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc2[r - 1]))
            logger.info("Validation Results RGB2IR ")
            logger.info("mAP: {:.1%}".format(mAP22))
            for r in [1, 5, 10, 20]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc22[r - 1]))

            logger.info("Validation Results IR2RGB ")
            logger.info("mAP: {:.1%}".format(mAP3))
            for r in [1, 5, 10, 20]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc3[r - 1]))
            logger.info("Validation Results RGB2IR ")
            logger.info("mAP: {:.1%}".format(mAP33))
            for r in [1, 5, 10, 20]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc33[r - 1]))
            # 2. evalloader_ir2rgb
            c0 = [cmc0[0], cmc1[0], cmc2[0], cmc3[0]]
            c1 = [cmc00[0], cmc11[0], cmc22[0], cmc33[0]]
            m0 = [mAP0, mAP1, mAP2, mAP3]
            m1 = [mAP00, mAP11, mAP22, mAP33]
            maxc0 = max(c0)
            maxc1 = max(c1)
            maxm0 = max(m0)
            maxm1 = max(m1)
            prec1 = maxc0 + maxc1 + maxm0 + maxm1
            is_best = prec1 > best_performance
            best_performance = max(prec1, best_performance)
            if is_best:
                best_epoch = epoch
            save_checkpoint(model.state_dict(), is_best, os.path.join(cfg.OUTPUT_DIR, 'checkpoint_ep.pth.tar'))

    logger.info("==> Best Perform {:.1%}, achieved at epoch {}".format(best_performance, best_epoch))
    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Total running time: {}".format(total_time))
    print(cfg.OUTPUT_DIR)


def do_inference_dense(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("TFCLIP.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        img = img.to(device)  # torch.Size([64, 4, 3, 256, 128])
        if len(img.size()) == 6:
            # method = 'dense'
            b, n, s, c, h, w = img.size()
            assert (b == 1)
            img = img.view(b * n, s, c, h, w)  # torch.Size([5, 8, 3, 256, 128])

        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else:
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else:
                target_view = None
            feat = model(img, cam_label=camids, view_label=target_view)
            feat = feat.view(-1, feat.size(1))
            feat = torch.mean(feat, 0, keepdim=True)  # 1,512
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)


    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]



def do_inference_rrs(cfg,
                     model,
                     evalloader_rgb2ir,
                     evalloader_ir2rgb,
                     num_query1,
                     num_query2
                     ):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator_rgb2ir = R1_mAP_eval(num_query1, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator_ir2rgb = R1_mAP_eval(num_query2, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator_rgb2ir.reset()
    evaluator_ir2rgb.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            # model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    ######### 1.rgb2ir eval
    for n_iter, (img, pid, camid, camids,) in enumerate(evalloader_rgb2ir):
        img = img.to(device)  # torch.Size([64, 4, 3, 256, 128])
        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)

            feat = model(img, cam_label=camids)
            # feat = feat.view(-1, feat.size(1))
            # feat = torch.mean(feat, 0, keepdim=True)  # 1,512
            evaluator_rgb2ir.update((feat, pid, camid))

    cmc0, mAP0, _, _, _, _, _ = evaluator_rgb2ir.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP0))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc0[r - 1]))

    ######### 1.rgb2ir eval
    for n_iter, (img, pid, camid, camids,) in enumerate(evalloader_ir2rgb):
        img = img.to(device)  # torch.Size([64, 4, 3, 256, 128])
        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else:
                camids = None
            feat = model(img, cam_label=camids)
            # feat = feat.view(-1, feat.size(1))
            # feat = torch.mean(feat, 0, keepdim=True)  # 1,512
            evaluator_ir2rgb.update((feat, pid, camid))
    cmc1, mAP1, _, _, _, _, _ = evaluator_ir2rgb.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP1))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc1[r - 1]))

    cmc = cmc0[0] + cmc1[0]
    return cmc[0]

def do_inference_dense_vcm(cfg,
                     model,
                     evalloader_rgb2ir,
                     num_query1,
                     num_query2
                     ):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator_rgb2ir = R1_mAP_eval(num_query1, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator_rgb2ir.reset()


    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            # model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    ######### 1.rgb2ir eval
    for n_iter, (img, pid, camid, camids,) in enumerate(evalloader_rgb2ir):
        img = img.to(device)  # torch.Size([64, 4, 3, 256, 128])
        if len(img.size()) == 6:
            # method = 'dense'
            b, n, s, c, h, w = img.size()
            assert (b == 1)
            img = img.view(b * n, s, c, h, w)  # torch.Size([5, 8, 3, 256, 128])
        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)

            out_feat_all, feat0, feat1, feat2 = model(img, cam_label=camids, eval_all=True)
            out_feat_all = out_feat_all.view(-1, out_feat_all.size(1))
            out_feat_all = torch.mean(out_feat_all, 0, keepdim=True)  # 1,512

            feat0 = feat0.view(-1, feat0.size(1))
            feat0 = torch.mean(feat0, 0, keepdim=True)  # 1,512

            feat1 = feat1.view(-1, feat1.size(1))
            feat1 = torch.mean(feat1, 0, keepdim=True)  # 1,512

            feat2 = feat2.view(-1, feat2.size(1))
            feat2 = torch.mean(feat2, 0, keepdim=True)  # 1,512


            evaluator_rgb2ir.update((out_feat_all, feat0, feat1, feat2, pid, camid))

    cmc0, mAP0, cmc00, mAP00, cmc1, mAP1, cmc11, mAP11, cmc2, mAP2, cmc22, mAP22, cmc3, mAP3, cmc33, mAP33 = evaluator_rgb2ir.compute()
    logger.info("Validation Results IR2RGB ")
    logger.info("mAP: {:.1%}".format(mAP0))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc0[r - 1]))
    logger.info("Validation Results RGB2IR ")
    logger.info("mAP: {:.1%}".format(mAP00))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc00[r - 1]))

    logger.info("Validation Results IR2RGB ")
    logger.info("mAP: {:.1%}".format(mAP1))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc1[r - 1]))
    logger.info("Validation Results RGB2IR ")
    logger.info("mAP: {:.1%}".format(mAP11))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc11[r - 1]))

    logger.info("Validation Results IR2RGB ")
    logger.info("mAP: {:.1%}".format(mAP2))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc2[r - 1]))
    logger.info("Validation Results RGB2IR ")
    logger.info("mAP: {:.1%}".format(mAP22))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc22[r - 1]))

    logger.info("Validation Results IR2RGB ")
    logger.info("mAP: {:.1%}".format(mAP3))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc3[r - 1]))
    logger.info("Validation Results RGB2IR ")
    logger.info("mAP: {:.1%}".format(mAP33))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc33[r - 1]))

    cmc = cmc0[0] + cmc1[0]
    return cmc[0]

def do_inference_rrs_vcm(cfg,
                     model,
                     evalloader_rgb2ir,
                     num_query1,
                     num_query2
                     ):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator_rgb2ir = R1_mAP_eval(num_query1, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator_rgb2ir.reset()


    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            # model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    ######### 1.rgb2ir eval
    for n_iter, (img, pid, camid, camids,) in enumerate(evalloader_rgb2ir):
        img = img.to(device)  # torch.Size([64, 4, 3, 256, 128])
        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)

            out_feat_all, feat0, feat1, feat2 = model(img, cam_label=camids)
            # feat = feat.view(-1, feat.size(1))
            # feat = torch.mean(feat, 0, keepdim=True)  # 1,512
            evaluator_rgb2ir.update((out_feat_all, feat0, feat1, feat2, pid, camid))

    cmc0, mAP0, cmc00, mAP00, cmc1, mAP1, cmc11, mAP11, cmc2, mAP2, cmc22, mAP22, cmc3, mAP3, cmc33, mAP33 = evaluator_rgb2ir.compute()
    logger.info("Validation Results IR2RGB ")
    logger.info("mAP: {:.1%}".format(mAP0))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc0[r - 1]))
    logger.info("Validation Results RGB2IR ")
    logger.info("mAP: {:.1%}".format(mAP00))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc00[r - 1]))

    logger.info("Validation Results IR2RGB ")
    logger.info("mAP: {:.1%}".format(mAP1))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc1[r - 1]))
    logger.info("Validation Results RGB2IR ")
    logger.info("mAP: {:.1%}".format(mAP11))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc11[r - 1]))

    logger.info("Validation Results IR2RGB ")
    logger.info("mAP: {:.1%}".format(mAP2))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc2[r - 1]))
    logger.info("Validation Results RGB2IR ")
    logger.info("mAP: {:.1%}".format(mAP22))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc22[r - 1]))

    logger.info("Validation Results IR2RGB ")
    logger.info("mAP: {:.1%}".format(mAP3))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc3[r - 1]))
    logger.info("Validation Results RGB2IR ")
    logger.info("mAP: {:.1%}".format(mAP33))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc33[r - 1]))

    cmc = cmc0[0] + cmc1[0]
    return cmc[0]