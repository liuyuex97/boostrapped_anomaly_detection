import numpy as np
import pickle, torch
from . import tools
import random


class Feeder_single(torch.utils.data.Dataset):
    """ Feeder for single inputs """

    def __init__(self, data_path, label_path, shear_amplitude=0.5, temperal_padding_ratio=6, mmap=True):
        self.data_path = data_path
        self.label_path = label_path

        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio
       
        self.load_data(mmap)

    def load_data(self, mmap):
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)
            # self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        #### for shanghai dataset
        data_numpy = np.zeros((3, 12, 18, 1))
        # print(np.array(self.data[index]).shape)
        data_numpy[:2, :, :, :] = np.array(self.data[index])#.transpose((3, 1, 2, 0))

        #### for kinetics dataset
        # loaded_data = np.array(self.data[index]).transpose((3, 1, 2, 0))
        # loaded_data = self.sampler(loaded_data)
        # data_numpy = np.zeros((3, 12, 18, 1))
        # data_numpy[:2, :, :, :] = loaded_data
        ####

        # data_numpy = np.array(self.data[index])
        label = self.label[index]
        sample_name = self.sample_name[index] # switch to load sample name instead

        # processing
        data = self._aug(data_numpy)[:2, :, :, :] # uncomment for shanghai
        return data, label, sample_name # sample_name

    def _aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)
        
        return data_numpy


class Feeder_dual(torch.utils.data.Dataset):
    """ Feeder for dual inputs """

    def __init__(self, data_path, label_path, shear_amplitude=0.5, temperal_padding_ratio=6, mmap=True):
        self.data_path = data_path
        self.label_path = label_path

        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio
       
        self.load_data(mmap)

    def load_data(self, mmap):
        # load label
        with open(self.label_path, 'rb') as f:
            # self.sample_name, self.label = pickle.load(f)
            # self.label = pickle.load(f)
            self.label = np.asarray(pickle.load(f)[1])
        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        # import pdb; pdb.set_trace()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        #### for shanghai dataset
        data_numpy = np.zeros((3, 12, 18, 1))
        data_numpy[:2, :, :, :] = np.array(self.data[index])[:2, :, :, :]

        #### for kinect dataset
        # import pdb; pdb.set_trace() # 240436, 3, 300, 18, 2
        # loaded_data = np.array(self.data[index]).transpose((1, 0, 2, 3))
        # loaded_data = self.sampler(loaded_data).transpose((1, 0, 2, 3))
        # data_numpy = np.zeros((3, 12, 18, 1)) # only use one person
        # if loaded_data[:, :, :, 1].sum() == 0.0:
        #     person_idx = 0
        # else:
        #     person_idx = random.choices([0, 1])
        # if len(loaded_data[:2, :, :, person_idx].shape) == 3:
        #     data_numpy[:2, :, :, :] = np.expand_dims(loaded_data[:2, :, :, person_idx], axis = 3)
        # else:
        #     data_numpy[:2, :, :, :] = loaded_data[:2, :, :, person_idx]
        ####

        label = self.label[index]
        
        # processing
        data1 = self._aug(data_numpy)[:2, :, :, :] #uncomment for shanghai dataset
        data2 = self._aug(data_numpy)[:2, :, :, :]

        ###### Mixup augmentation #######
        mixup_idx = random.randint(0, len(self.data)-1)
        mixup_label = self.label[mixup_idx]
        mixup_data = np.array(self.data[mixup_idx])#.transpose((1, 0, 2, 3))

        ##### kinetics ####
        # mixup_data = self.sampler(mixup_data).transpose((1, 0, 2, 3))
        # if mixup_data[:, :, :, 1].sum() == 0.0:
        #     person_idx = 0
        # else:
        #     person_idx = random.choices([0, 1])
        # if len(mixup_data[:2, :, :, person_idx].shape) == 3:
        #     mixup_data = np.expand_dims(mixup_data[:2, :, :, person_idx], axis = 3)
        # else:
        #     mixup_data = mixup_data[:2, :, :, person_idx]
        #####
        # Select a random number from the given beta distribution
        # Mixup the images accordingly
        alpha = 0.2
        lam = np.random.beta(alpha, alpha)
        data3 = lam * data1 + (1 - lam) * mixup_data
        data4 = np.zeros((3, 12, 18, 1))
        data4[:2, :, :, :] = data3[:2, :, :, :]
        data4 = self._aug(data4)[:2, :, :, :]
        label = lam * label + (1 - lam) * mixup_label

        return [data1, data2, data3, data4], label

    def _aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)
        
        return data_numpy

    def sampler(self, data_numpy, block_len=12):
        vlen = data_numpy.shape[0]
        self.block1_start_pos = "anywhere"
        self.num_pretrain_blocks = 1
        self.downsample = 1
        self.num_blocks = 1
        if self.block1_start_pos == "anywhere":
            block1_start_idx = np.random.choice(
                    range(0, vlen-self.num_pretrain_blocks*block_len*self.downsample), 1).item()
        elif self.block1_start_pos == "zero":
            block1_start_idx = 0
        idx_blocks = np.array([[block1_start_idx]])+np.expand_dims(
                np.arange(block_len), 0)*self.downsample
        idx_blocks = idx_blocks.reshape(self.num_blocks*block_len)
        frames = data_numpy[idx_blocks]
        return frames



# class Feeder_semi(torch.utils.data.Dataset):
#     """ Feeder for semi-supervised learning """

#     def __init__(self, data_path, label_path, shear_amplitude=0.5, temperal_padding_ratio=6, mmap=True, label_list=None):
#         self.data_path = data_path
#         self.label_path = label_path

#         self.shear_amplitude = shear_amplitude
#         self.temperal_padding_ratio = temperal_padding_ratio
#         self.label_list = label_list
       
#         self.load_data(mmap)
#         self.load_semi_data()    

#     def load_data(self, mmap):
#         # load label
#         with open(self.label_path, 'rb') as f:
#             self.sample_name, self.label = pickle.load(f)

#         # load data
#         if mmap:
#             self.data = np.load(self.data_path, mmap_mode='r')
#         else:
#             self.data = np.load(self.data_path)

#     def load_semi_data(self):
#         data_length = len(self.label)

#         if not self.label_list:
#             self.label_list = list(range(data_length))
#         else:
#             self.label_list = np.load(self.label_list).tolist()
#             self.label_list.sort()

#         self.unlabel_list = list(range(data_length))

#     def __len__(self):
#         return len(self.unlabel_list)

#     def __getitem__(self, index):
#         # get data
#         data_numpy = np.array(self.data[index])
#         label = self.label[index]
        
#         # processing
#         data = self._aug(data_numpy)
#         return data, label
    
#     def __getitem__(self, index):
#         label_index = self.label_list[index % len(self.label_list)]
#         unlabel_index = self.unlabel_list[index]

#         # get data
#         label_data_numpy = np.array(self.data[label_index])
#         unlabel_data_numpy = np.array(self.data[unlabel_index])
#         label = self.label[label_index]
        
#         # processing
#         data1 = self._aug(unlabel_data_numpy)
#         data2 = self._aug(unlabel_data_numpy)
#         return [data1, data2], label_data_numpy, label

#     def _aug(self, data_numpy):
#         if self.temperal_padding_ratio > 0:
#             data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

#         if self.shear_amplitude > 0:
#             data_numpy = tools.shear(data_numpy, self.shear_amplitude)
        
#         return data_numpy
