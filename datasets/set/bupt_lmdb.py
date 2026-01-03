from __future__ import print_function, absolute_import
import os
import os.path as osp
import numpy as np
import json
from os.path import join
import lmdb
import pickle

'''
def decoder_pic_path(fname):
    base = fname[0:4]
    modality = fname[5]
    if modality == '1' :
        modality_str = 'ir'
    else:
        modality_str = 'rgb'
    camera = fname[6:8]
    F_pos = fname.find('F')
    picture = fname[F_pos+1:]
    path = base + '/' + modality_str + '/' + camera + '/' + picture
    return path
'''
MODALITY = {'RGB/IR': -1, 'RGB': 0, 'IR': 1}
MODALITY_ = {-1: 'All', 0: 'RGB', 1: 'IR'}
CAMERA = {'LS3': 0, 'G25': 1, 'CQ1': 2, 'W4': 3, 'TSG1': 4, 'TSG2': 5}


def decoder_pic_path(fname):
    base = fname[0:4]
    modality = fname[5]
    if modality == '1':
        modality_str = 'ir'
    else:
        modality_str = 'rgb'

    # D/T/F identify a certain frame
    # D=camera id
    # F=frame id
    # T=tracklet id

    T_pos = fname.find('T')
    D_pos = fname.find('D')
    F_pos = fname.find('F')
    camera = fname[D_pos:T_pos]
    picture = fname[F_pos + 1:]
    path = base + '/' + modality_str + '/' + camera + '/' + picture
    return path


class BUPT_lmdb(object):
    root1 = '/13994058190/YCY/dataset/bupt_lmdb/'
    root2 = '/media/ycy/ba8af05f-f397-4839-a318-f469b124cbab/data/BUPT_lmdb/'
    root3 = '/18640539002/dataset/bupt_lmdb/'

    if os.path.exists(root1):
        root = root1
    elif os.path.exists(root2):
        root = root2
    elif os.path.exists(root3):
        root = root3
    # training data
    lmdb_train_ir = osp.join(root, 'train_ir.lmdb')
    lmdb_train_rgb = osp.join(root, 'train_rgb.lmdb')

    lmdb_query_ir = osp.join(root, 'query_ir.lmdb')
    lmdb_query_rgb = osp.join(root, 'query_rgb.lmdb')

    lmdb_gallery_ir = osp.join(root, 'gallery_ir.lmdb')
    lmdb_gallery_rgb = osp.join(root, 'gallery_rgb.lmdb')

    def __init__(self, min_seq_len=12):
        self._check_before_run()

        # 1.read lmdb
        self.env_train_ir = lmdb.open(self.lmdb_train_ir,
                                      subdir=os.path.isdir(self.lmdb_train_ir),
                                      readonly=True, lock=False,
                                      readahead=False, meminit=False)
        self.env_train_rgb = lmdb.open(self.lmdb_train_rgb,
                                       subdir=os.path.isdir(self.lmdb_train_rgb),
                                       readonly=True, lock=False,
                                       readahead=False, meminit=False)

        self.env_query_ir = lmdb.open(self.lmdb_query_ir,
                                      subdir=os.path.isdir(self.lmdb_query_ir),
                                      readonly=True, lock=False,
                                      readahead=False, meminit=False)
        self.env_query_rgb = lmdb.open(self.lmdb_query_rgb,
                                       subdir=os.path.isdir(self.lmdb_query_rgb),
                                       readonly=True, lock=False,
                                       readahead=False, meminit=False)

        self.env_gallery_ir = lmdb.open(self.lmdb_gallery_ir,
                                        subdir=os.path.isdir(self.lmdb_gallery_ir),
                                        readonly=True, lock=False,
                                        readahead=False, meminit=False)
        self.env_gallery_rgb = lmdb.open(self.lmdb_gallery_rgb,
                                         subdir=os.path.isdir(self.lmdb_gallery_rgb),
                                         readonly=True, lock=False,
                                         readahead=False, meminit=False)
        train_ir, num_train_tracklets_ir, num_train_imgs_ir, num_train_pids_ir, ir_label = self._process_data_lmdb(
            self.env_train_ir)

        train_rgb, num_train_tracklets_rgb, num_train_imgs_rgb, num_train_pids_rgb, rgb_label = self._process_data_lmdb(
            self.env_train_rgb)

        query_ir, num_query_tracklets_ir, num_query_imgs_ir, num_query_pids_ir, query_ir_label = self._process_data_lmdb(
            self.env_query_ir)
        query_rgb, num_query_tracklets_rgb, num_query_imgs_rgb, num_query_pids_rgb, query_rgb_label = self._process_data_lmdb(
            self.env_query_rgb)
        gallery_ir, num_gallery_tracklets_ir, num_gallery_imgs_ir, num_gallery_pids_ir, gallery_ir_label = self._process_data_lmdb(
            self.env_gallery_ir)
        gallery_rgb, num_gallery_tracklets_rgb, num_gallery_imgs_rgb, num_gallery_pids_rgb, gallery_rgb_label = self._process_data_lmdb(
            self.env_gallery_rgb)

        # ---------------------------------------

        print("=> VCM loaded")
        print("Dataset statistics:")
        print("---------------------------------")
        print("subset      | # ids | # tracklets")
        print("---------------------------------")
        print("train_ir    | {:5d} | {:8d}".format(num_train_pids_ir, num_train_tracklets_ir))
        print("train_rgb   | {:5d} | {:8d}".format(num_train_pids_rgb, num_train_tracklets_rgb))
        print("query_IR/RGB       | {:5d} | {:8d}".format(num_query_tracklets_ir, num_query_tracklets_rgb))
        print("gallery_IR/RGB     | {:5d} | {:8d}".format(num_gallery_tracklets_ir, num_gallery_tracklets_rgb))

        print("---------------------------------")
        self.num_train_cams = 6
        self.train_ir = train_ir  # 3574
        self.train_rgb = train_rgb  # 3574
        self.ir_label = ir_label
        self.rgb_label = rgb_label

        self.query = query_ir  # 536
        self.gallery = gallery_rgb  # 2422

        self.num_train_pids = num_train_pids_ir  # 1074

        self.num_query_tracklets = num_query_tracklets_ir  # 536
        self.num_gallery_tracklets = num_gallery_tracklets_rgb  # 2422

        # ------- visible to infrared------------
        self.query_1 = query_rgb  # 540
        self.gallery_1 = gallery_ir  # 2422
        self.num_train_tracklets_ir = num_train_tracklets_ir
        self.num_train_tracklets_rgb = num_train_tracklets_rgb

        # self.num_gallery_pids_1 = num_gallery_pids_1
        self.num_query_tracklets_1 = num_query_tracklets_rgb
        self.num_gallery_tracklets_1 = num_gallery_tracklets_ir
        # ---------------------------------------

    def _parse_data(self, path):
        data_info, pids = [], []
        # path = join(self.data_root, path)
        with open(path) as f:
            for line in f.readlines():
                obj_id, modality, camera, tracklet_id = line.strip().split(' ')
                data_info.append((obj_id, modality, camera, tracklet_id))
                pids.append(obj_id)
        # self.pids = sorted(set(pids))
        return data_info, pids

    def _parse_train_data(self, path):
        data_info, pids = [], []
        # path = join(self.data_root, path)
        with open(path) as f:
            for line in f.readlines():
                obj_id, modality, camera, tracklet_id = line.strip().split(' ')
                data_info.append((obj_id, modality, camera, tracklet_id))
                pids.append(obj_id)
        # self.pids = sorted(set(pids))
        return data_info, pids

    def _check_before_run(self):
        """check before run """
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.lmdb_train_ir):
            raise RuntimeError("'{}' is not available".format(self.lmdb_train_ir))
        if not osp.exists(self.lmdb_train_rgb):
            raise RuntimeError("'{}' is not available".format(self.lmdb_train_rgb))
        if not osp.exists(self.lmdb_query_ir):
            raise RuntimeError("'{}' is not available".format(self.lmdb_query_ir))
        if not osp.exists(self.lmdb_query_rgb):
            raise RuntimeError("'{}' is not available".format(self.lmdb_query_rgb))
        if not osp.exists(self.lmdb_gallery_ir):
            raise RuntimeError("'{}' is not available".format(self.lmdb_gallery_ir))
        if not osp.exists(self.lmdb_gallery_rgb):
            raise RuntimeError("'{}' is not available".format(self.lmdb_gallery_rgb))

    def _get_names(self, fpath):
        """get image name, retuen name list"""
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def _get_tracks(self, fpath):
        """get tracks file"""
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                new_line.split(' ')

                tmp = new_line.split(' ')[0:]

                tmp = list(map(int, tmp))
                names.append(tmp)
        names = np.array(names)
        return names

    '''
    def _get_query_idx(self,fpath):
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                new_line.split(' ')
                # print(new_line.split(' ')[0:-1])
                tmp = new_line.split(' ')[0:-1]
                if new_line.split(' ')[-1][-1] == '1':
                    tmp.append(new_line.split(' ')[-1][0] + '1')
                else:
                    tmp.append(new_line.split(' ')[-1][0] + '2')
                tmp = list(map(int, tmp))
                idxs = tmp
        idxs = np.array(idxs)
        print(idxs)
        return idxs
    '''

    def _get_query_idx(self, fpath):
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                new_line.split(' ')
                tmp = new_line.split(' ')[0:]

                tmp = list(map(int, tmp))
                idxs = tmp
        idxs = np.array(idxs)
        # print(idxs)
        return idxs

    def _process_data_lmdb(self, env):
        # meta data: tracklets list
        # every tracklet = [modality label, start frame id, end frame id, pid, camid]
        # "start frame id" starts from 1 and increases by 24 each time until 232458.

        tracklets_ir = []
        num_imgs_per_tracklet_ir = []
        ir_label = []
        label = []

        with env.begin() as txn:
            length = pickle.loads(txn.get(b'__len__'))
            keys = pickle.loads(txn.get(b'__keys__'))

            for i in range(length):
                byteflow = txn.get(keys[i])
                IMAGE = pickle.loads(byteflow)
                img_paths, pid, camid = IMAGE[0], IMAGE[1], IMAGE[2]
                # if len(img_paths) == 1:
                #     continue
                num_imgs_per_tracklet_ir.append(len(img_paths))
                ir_label.append(pid)
                label.append(pid)
                tracklets_ir.append((img_paths, pid, camid))

        num_pids = len(set(label))
        num_tracklets_ir = len(tracklets_ir)  # 3049

        return tracklets_ir, num_tracklets_ir, num_imgs_per_tracklet_ir, num_pids, ir_label,

    def _process_data_query(self, query_info, query_pids, relabel=False, min_seq_len=0):
        # meta data: tracklets list
        # every tracklet = [modality label, start frame id, end frame id, pid, camid]
        # "start frame id" starts from 1 and increases by 24 each time until 232458.

        num_tracklets = len(query_info)  # 1076
        pid_list = list(set(query_pids))
        num_pids = len(pid_list)  # 1074

        # if relabel: pid2label = {pid: label for label, pid in enumerate(pid_list)}

        tracklets_ir = []
        num_imgs_per_tracklet_ir = []
        ir_label = []

        tracklets_rgb = []
        num_imgs_per_tracklet_rgb = []
        rgb_label = []

        for tracklet_idx in range(num_tracklets):
            data = query_info[tracklet_idx]
            obj_id, modality, camera, tracklet_id = data
            # if relabel: obj_id, = pid2label[obj_id]
            # mm.append(modality)
            if modality == 'IR':  # IR modality
                # The "names" stores information for 230405 images in the format of [pid,modality label,cam id, tracklet id, image id]
                img_ir_names = self.data_paths[obj_id]['IR'][camera][tracklet_id]
                img_ir_paths = [osp.join(self.DATA_path, img_name_ir) for img_name_ir in img_ir_names]
                # if len(img_ir_paths) >= min_seq_len:  # Filter out samples with low frame rates
                img_ir_paths = tuple(img_ir_paths)  # The paths of all frames in a tracklet

                ir_label.append(int(obj_id))
                camid = CAMERA[camera]
                tracklets_ir.append((img_ir_paths, int(obj_id), camid))
                # same id
                num_imgs_per_tracklet_ir.append(len(img_ir_paths))

            elif modality == 'RGB':  # IR modality
                img_rgb_names = self.data_paths[obj_id]['RGB'][camera][tracklet_id]
                img_rgb_paths = [osp.join(self.DATA_path, img_name_rgb) for img_name_rgb in img_rgb_names]

                # if len(img_rgb_paths) >= min_seq_len:
                img_rgb_paths = tuple(img_rgb_paths)
                rgb_label.append(int(obj_id))
                camid = CAMERA[camera]
                tracklets_rgb.append((img_rgb_paths, int(obj_id), camid))
                # same id
                num_imgs_per_tracklet_rgb.append(len(img_rgb_paths))
            else:
                raise RuntimeError('Only modality RGB/IR is supported for test.')

        num_tracklets_ir = len(tracklets_ir)  # 3049   2422
        num_tracklets_rgb = len(tracklets_rgb)  # 3049  2422

        num_tracklets = num_tracklets_rgb + num_tracklets_ir  # 6098

        return tracklets_ir, num_tracklets_ir, num_imgs_per_tracklet_ir, tracklets_rgb, num_tracklets_rgb, num_imgs_per_tracklet_rgb, num_pids, ir_label, rgb_label

    def _process_data_test(self, names, meta_data, relabel=False, min_seq_len=0):
        # meta_data format
        # [1 284 307 503 4]
        # 0: modality id
        # 1: start frame index
        # 2: end frame index
        # 3: pid
        # 4: camera id
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:, 3].tolist()))
        num_pids = len(pid_list)

        if relabel: pid2label = {pid: label for label, pid in enumerate(pid_list)}
        tracklets = []
        num_imgs_per_tracklet = []

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx, ...]
            m, start_index, end_index, pid, camid = data
            if relabel: pid = pid2label[pid]

            img_names = names[start_index - 1:end_index]
            img_paths = [osp.join(self.root, "Test", decoder_pic_path(img_name)) for img_name in img_names]
            if len(img_paths) >= min_seq_len:
                img_paths = tuple(img_paths)
                tracklets.append((img_paths, pid, camid))
                num_imgs_per_tracklet.append(len(img_paths))

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet


if __name__ == '__main__':
    dataset = BUPT_lmdb()
