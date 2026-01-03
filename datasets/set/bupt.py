from __future__ import print_function, absolute_import
import os
import os.path as osp
import numpy as np
import json
from os.path import join

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


class BUPT(object):
    root1 = '/13994058190/YCY/dataset/BUPT/'
    root2 = '/media/ycy/ba8af05f-f397-4839-a318-f469b124cbab/data/BUPT/'
    root3 = '/18640539002/dataset/BUPT/'

    if os.path.exists(root1):
        root = root1
    elif os.path.exists(root2):
        root = root2
    elif os.path.exists(root3):
        root = root3
    # training data
    DATA_path = osp.join(root, 'DATA')
    track_all_info_json_path = osp.join(root, 'data_paths.json')
    track_train_txt_path = osp.join(root, 'train.txt')
    track_train_aux_txt_path = osp.join(root, 'train_auxiliary.txt')

    # testing data
    track_query_txt_path = osp.join(root, 'query.txt')
    track_gallery_txt_path = osp.join(root, 'gallery.txt')

    def __init__(self, min_seq_len=12):
        self._check_before_run()

        # prepare meta data
        self.data_paths = json.load(open(self.track_all_info_json_path))  # 3080
        train_info, train_pids = self._parse_train_data(
            self.track_train_txt_path)  # 3574  ('1746', 'RGB/IR', 'TSG2', '1')
        train_aux_info, _ = self._parse_data(self.track_train_aux_txt_path)  # 930  ('2830', 'RGB/IR', 'TSG1', '1')
        query_info, query_pids = self._parse_data(self.track_query_txt_path)  # 1076  ('1911', 'RGB', 'G25', '1')
        gallery_info, gallery_pids = self._parse_data(self.track_gallery_txt_path)  # 4844  ('1911', 'IR', 'LS3', '1')

        train_ir, num_train_tracklets_ir, num_train_imgs_ir, train_rgb, num_train_tracklets_rgb, num_train_imgs_rgb, num_train_pids, ir_label, rgb_label = \
            self._process_data_train(train_info, train_pids, relabel=True, min_seq_len=min_seq_len)

        query_ir, num_query_tracklets_ir, num_query_imgs_ir, query_rgb, num_query_tracklets_rgb, num_query_imgs_rgb, num_query_pids, query_ir_label, query_rgb_label = \
            self._process_data_query(query_info, query_pids, relabel=False, min_seq_len=min_seq_len)

        gallery_ir, num_gallery_tracklets_ir, num_gallery_imgs_ir, gallery_rgb, num_gallery_tracklets_rgb, num_gallery_imgs_rgb, num_gallery_pids, gallery_ir_label, gallery_rgb_label = \
            self._process_data_query(gallery_info, gallery_pids, relabel=False, min_seq_len=min_seq_len)

        num_imgs_per_tracklet = num_train_imgs_rgb + num_gallery_imgs_ir + num_query_imgs_ir + num_query_imgs_rgb + num_gallery_imgs_ir + num_gallery_imgs_rgb
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)
        # ---------------------------------------

        print("=> VCM loaded")
        print("Dataset statistics:")
        print("---------------------------------")
        print("subset      | # ids | # tracklets")
        print("---------------------------------")
        print("train_ir    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets_ir))
        print("train_rgb   | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets_rgb))
        print("query_IR/RGB       | {:5d} | {:8d}".format(num_query_tracklets_ir, num_query_tracklets_rgb))
        print("gallery_IR/RGB     | {:5d} | {:8d}".format(num_gallery_tracklets_ir, num_gallery_tracklets_rgb))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("---------------------------------")
        self.num_train_cams = 6
        self.train_ir = train_ir  # 3574
        self.train_rgb = train_rgb  # 3574
        self.ir_label = ir_label
        self.rgb_label = rgb_label

        self.query = query_ir  # 536
        self.gallery = gallery_rgb  # 2422

        self.num_train_pids = num_train_pids  # 1074
        self.num_query_pids = num_query_pids  # 1076
        self.num_gallery_pids = num_gallery_pids
        self.num_query_tracklets = num_query_tracklets_ir  # 536
        self.num_gallery_tracklets = num_gallery_tracklets_rgb  # 2422

        # ------- visible to infrared------------
        self.query_1 = query_rgb  # 540
        self.gallery_1 = gallery_ir  # 2422
        self.num_train_tracklets_ir = num_train_tracklets_ir
        self.num_train_tracklets_rgb = num_train_tracklets_rgb

        self.num_query_pids_1 = num_query_pids
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
        if not osp.exists(self.DATA_path):
            raise RuntimeError("'{}' is not available".format(self.DATA_path))
        if not osp.exists(self.track_all_info_json_path):
            raise RuntimeError("'{}' is not available".format(self.track_all_info_json_path))
        if not osp.exists(self.track_train_txt_path):
            raise RuntimeError("'{}' is not available".format(self.track_train_txt_path))
        if not osp.exists(self.track_train_aux_txt_path):
            raise RuntimeError("'{}' is not available".format(self.track_train_aux_txt_path))
        if not osp.exists(self.track_query_txt_path):
            raise RuntimeError("'{}' is not available".format(self.track_query_txt_path))
        if not osp.exists(self.track_gallery_txt_path):
            raise RuntimeError("'{}' is not available".format(self.track_gallery_txt_path))

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

    def _process_data_train(self, train_info, train_pids, relabel=False, min_seq_len=0):
        # meta data: tracklets list
        # every tracklet = [modality label, start frame id, end frame id, pid, camid]
        # "start frame id" starts from 1 and increases by 24 each time until 232458.

        num_tracklets = len(train_info)  # 3574
        pid_list = list(set(train_pids))
        num_pids = len(pid_list)  # 1074

        if relabel: pid2label = {pid: label for label, pid in enumerate(pid_list)}

        tracklets_ir = []
        num_imgs_per_tracklet_ir = []
        ir_label = []

        tracklets_rgb = []
        num_imgs_per_tracklet_rgb = []
        rgb_label = []

        for tracklet_idx in range(num_tracklets):
            data = train_info[tracklet_idx]
            obj_id, modality, camera, tracklet_id = data
            if relabel: pid = pid2label[obj_id]

            if modality == 'RGB/IR':  # IR modality
                # The "names" stores information for 230405 images in the format of [pid,modality label,cam id, tracklet id, image id]
                img_ir_names = self.data_paths[obj_id]['IR'][camera][tracklet_id]
                img_rgb_names = self.data_paths[obj_id]['RGB'][camera][tracklet_id]
                img_ir_paths = [osp.join(self.DATA_path, img_name_ir) for img_name_ir in img_ir_names]
                img_rgb_paths = [osp.join(self.DATA_path, img_name_rgb) for img_name_rgb in img_rgb_names]
                # if len(img_ir_paths) >= min_seq_len:  # Filter out samples with low frame rates
                img_ir_paths = tuple(img_ir_paths)  # The paths of all frames in a tracklet
                if len(img_ir_paths) == 1:
                    continue

                ir_label.append(pid)
                camid = CAMERA[camera]
                tracklets_ir.append((img_ir_paths, pid, camid))

                # same id
                num_imgs_per_tracklet_ir.append(len(img_ir_paths))

                # if len(img_rgb_paths) >= min_seq_len:
                img_rgb_paths = tuple(img_rgb_paths)
                rgb_label.append(pid)
                camid = CAMERA[camera]
                tracklets_rgb.append((img_rgb_paths, pid, camid))
                # same id
                num_imgs_per_tracklet_rgb.append(len(img_rgb_paths))
            else:
                raise RuntimeError('Only modality RGB/IR is supported for training.')

        num_tracklets_ir = len(tracklets_ir)  # 3049
        num_tracklets_rgb = len(tracklets_rgb)  # 3049

        num_tracklets = num_tracklets_rgb + num_tracklets_ir  # 6098

        return tracklets_ir, num_tracklets_ir, num_imgs_per_tracklet_ir, tracklets_rgb, num_tracklets_rgb, num_imgs_per_tracklet_rgb, num_pids, ir_label, rgb_label

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
                if len(img_ir_paths) == 1:
                    continue
                tracklets_ir.append((img_ir_paths, int(obj_id), camid))
                # same id
                num_imgs_per_tracklet_ir.append(len(img_ir_paths))

            elif modality == 'RGB':  # IR modality
                img_rgb_names = self.data_paths[obj_id]['RGB'][camera][tracklet_id]
                img_rgb_paths = [osp.join(self.DATA_path, img_name_rgb) for img_name_rgb in img_rgb_names]
                # if len(img_rgb_paths) >= min_seq_len:
                img_rgb_paths = tuple(img_rgb_paths)
                if len(img_rgb_paths) == 1:
                    continue
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


if __name__ == '__main__':
    dataset = BUPT()
