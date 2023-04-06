# phase 2

from torch.utils.data.dataset import Dataset
import os
import numpy as np
import cv2
import torch
import pickle
import pdb

#this dataloader would be used for fixed view mapping


class NTUDataset(Dataset):
    """NTU Dataset"""

    def __init__(self, root_dir, data_file,dict_file=None ,param_file=None ,caption_file = "./caption.txt",
                 resize_height=64, resize_width=64, clip_len=16,
                 height=135, width=240, skip_len=2,
                 view1=1, view2=2, random_views=False, random_all=False, inceptionv3_norm = False,
                 precrop=False):
        """
        Initializes the NTU Dataset object used for extracting samples to run through the network.
        :param root_dir: (str) Directory with all the frames extracted from the videos.
        :param data_file: (str) Path to .txt file containing the sample IDs.
        :param resize_height: (int) The desired frame height.
        :param resize_width: (int) The desired frame width.
        :param clip_len: (int) The number of frames desired in the sample clip.
        :param height: (int, optional) The height of the frames in the dataset (default 135 for NTU Dataset).
        :param width: (int, optional) The width of the frames in the dataset (default 240 for NTU Dataset).
        :param skip_len: (int, optional) The number of frames to skip between each when creating the clip (default 1).
        :param view1: (int, optional) The desired viewpoint to use as the first view; can be 1, 2, or 3 (default 1).
        :param view2: (int, optional) The desired viewpoint to use as the seconds view; can be 1, 2, or 3 (default 2).
        :param random_views: (boolean, optional) True to use 2 constant randomly generated views (default False).
        :param random_all: (boolean, optional) True to use 2 randomly generated views for each sample (default False).
        :param precrop: (boolean, optional) True to crop 50 pixels off left and right of frame before randomly cropping
                        (default False).
        """
        with open(data_file) as f:
            self.data_file = f.readlines()
        self.data_file = [line.strip() for line in self.data_file]
        with open(caption_file) as f:
            self.caption_file = f.readlines()
        self.caption_file = [line.strip() for line in self.caption_file]
        self.root_dir = root_dir
        self.dict_file = dict_file
        self.param_file = param_file
        self.clip_len = clip_len
        self.skip_len = skip_len
        self.height = height
        self.width = width
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.channels = 3
        self.view1 = view1
        self.view2 = view2
        if random_views:
            self.get_random_views()
        self.random_all = random_all
        self.precrop = precrop
        
        self.mean = (0.5, 0.5, 0.5)
        self.std = (0.5, 0.5, 0.5)
        if inceptionv3_norm:
            print("Using Inception V3 norm")
            self.mean = (0.485, 0.456, 0.406)
            self.std = (0.229, 0.224, 0.225)
#         self.view_params = self.load_view_params()

    def load_view_params(self):
        """
        Function to load the params associated with the camera view for the NTU Dataset.
        :return: A
        """
        view_params = np.loadtxt(self.param_file)

        # normalize the distances
        view_params /= view_params.max(axis=0)

        return view_params

    def __len__(self):
        """
        Function to return the number of samples in the dataset.
        :return: int representing the number of samples in the dataset.
        """
        return len(self.data_file)

    def __getitem__(self, idx):
        """
        Function to get a single sample from the dataset. All modifications to the sample are done here.
        :param idx: (int) The index of the sample to get.
        :return: a float representing the viewpoint difference from the perspective of the first view, i.e. a negative
                 value indicates that the second view is clockwise from the first and a positive values indicates that
                 the second value is counterclockwise from the first,
                 and 2 tensors representing the sample video from 2 viewpoints
        """
        if self.random_all:
            self.get_random_views()

        action, sample_id, nf_v1= self.process_index(index=idx)
        
#         idx2 = self.get_video2_idx(index=idx,action=action)
#         action2, sample_id2, nf_v2 = self.process_index(index=idx2)
        
        view1path = self.get_vid_paths(action=action, sample_id=sample_id)
        #view2path = self.get_vid_paths(action=action, sample_id=sample_id2)

        # vp_diff = vp2 - vp1
#         caption = self.get_caption(action)

        frame_count = nf_v1
        frame_index = self.rand_frame_index(frame_count=frame_count)
        #pixel_index = self.rand_pixel_index()

        vid1 = self.load_frames(vid_path=view1path,
                                frame_index=frame_index, pixel_index=[0,0])
        #vid2 = self.load_frames(vid_path=view2path,
        #                        frame_index=frame_index, pixel_index=pixel_index)
        vid1 = NTUDataset.to_tensor(sample=vid1)
        vid1 = NTUDataset.normalize_clip_local(vid1, self.mean, self.std)
        #pdb.set_trace()
        # clip is vid1, target is action
        # need to apply same transformations to clip.
        return  action, vid1, frame_index, nf_v1 #, (action-1)
        
    def get_video2_idx(self, index, action):
        """
        this returns the details of the random video of same action which is present
         in mapping.pickle 
        """
        pickle_in = open(self.dict_file,'rb')
        vid_dict = pickle.load(pickle_in)
        vid2_idx = vid_dict[int(action)][self.data_file[index]]
#         info, action, id, action_no = self.process_index(index)
        return vid2_idx


    def get_random_views(self):
        """
        Function to generate 2 randomStuff viewpoints for the sample.
        :return: 2 ints representing the viewpoints for the sample.
        """
        self.view1, self.view2 = np.random.randint(1, 4), np.random.randint(1, 4)
        while self.view2 == self.view1:
            self.view2 = np.random.randint(1, 4)

    def process_index(self, index):
        """
        Function to process the information that the data file contains about the sample.
        The line of information contains the sample name as well as the number of available frames from each view
        :param index: (int) The index of the sample.
        :return: the action class, sample_id, two floats representing the viewpoint angles, and two ints representing
                 the number of frames in view1 and view2
        """
        sample_info = self.data_file[index].split(' ')
        sample_id = sample_info[0][sample_info[0].index('/') + 1:]
        scene, pid, rid, action = NTUDataset.decrypt_vid_name(vid_name=sample_id)
        
#         pickle_in = open(self.dict_file,'rb')
#         vid_dict = pickle.load(pickle_in)
#         vid2_idx = vid_dict[int(action)][self.data_file[index]]
#         sample_info2 = vid2_idx.split(' ')
#         sample_id2 = sample_info2[0][sample_info2[0].index('/') + 1:]
#         scene2, pid2, rid2, action2 = NTUDataset.decrypt_vid_name(vid_name=sample_id2)

# #         angle_v1 = NTUDataset.get_viewing_angle(rid=rid, cam=self.view1)
#         angle_v2 = NTUDataset.get_viewing_angle(rid=rid, cam=self.view2)

        nf_v1, nf_v2, nf_v3 = sample_info[1:]
        nf_v1, nf_v2, nf_v3 = int(nf_v1), int(nf_v2), int(nf_v3)

        info = (action, sample_id)
        if self.view1 == 1:
            info = info + (nf_v1,)
        elif self.view1 == 2:
            info = info + (nf_v2,)
        elif self.view1 == 3:
            info = info + (nf_v3,)
            
#         nf_v1, nf_v2, nf_v3 = sample_info2[1:]
#         nf_v1, nf_v2, nf_v3 = int(nf_v1), int(nf_v2), int(nf_v3)
#         if self.view1 == 1:
#             info = info + (nf_v1,)
#         elif self.view1 == 2:
#             info = info + (nf_v2,)
#         elif self.view1 == 3:
#             info = info + (nf_v3,)
            
        return info

    def get_caption(self,action):
        """
        This will take as input the action number and return the caption for that action
        """

        return self.caption_file[int(action)-1]


    @staticmethod
    def get_viewing_angle(rid, cam):
        """
        Function to get the camera viewing angles.
        :param rid: (int) The repetition ID of the sample video. Legal values are 1 and 2.
        :param cam: (int) The camera ID. Legal values are 1, 2, and 3.
        :return: float representing the camera viewing angle.
        """
        vpt = 0.
        pi = 22 / 7.
        # rid-1 implies face towards cam3; rid-2 implies face towards cam2; cam1 is the center camera
        if rid == 1:
            if cam == 1:
                vpt = pi / 4
            elif cam == 2:
                vpt = pi / 2
            elif cam == 3:
                vpt = 0.00
        elif rid == 2:
            if cam == 1:
                vpt = -pi / 4
            elif cam == 2:
                vpt = 0.00
            elif cam == 3:
                vpt = -pi / 2

        return vpt

    def get_vid_paths(self, action, sample_id):
        """
        Function to get the paths at which the two sample views are located.
        :param action: (int) The action class that the sample captures.
        :param sample_id: (str) The id for the sample from the data file.
        :return: 2 strings representing the paths for the sample views.
        """
        vid_path = self.make_sample_path(action=str(action), sample_id=sample_id)

        view1_path = os.path.join(vid_path, str(self.view1))

        return view1_path

    def make_sample_path(self, action, sample_id):
        """
        Function to make the path at which the sample is located.
        :param action: (int) The action class that the sample captures.
        :param sample_id: (str) The id for the sample from the data file.
        :return: The path at which the sample is located.
        """
        vid_path = os.path.join(self.root_dir, str(action), sample_id)

        return vid_path

    @staticmethod
    def decrypt_vid_name(vid_name):
        """
        Function to break up the meaning of the video name.
        :param vid_name: (string) The name of the video.
        :return: 4 ints representing the scene, person, repetition, and action that the video captures.
        """
        scene = int(vid_name[1:4])
        pid = int(vid_name[5:8])
        rid = int(vid_name[9:12])
        action = int(vid_name[13:16])

        return scene, pid, rid, action

    def rand_frame_index(self, frame_count):
        """
        Function to generate a randomStuff starting frame index for cropping the temporal dimension of the video.
        :param frame_count: (int) The number of available frames in the sample video.
        :return: The starting frame index for the sample.
        """
        max_frame = frame_count - (self.skip_len * self.clip_len)
        if frame_count > 64:
            frame_index = np.random.randint(frame_count/6, frame_count/2)
        else:
            frame_index = np.random.randint(0, max_frame)
#             frame_index = np.random.randint(frame_count/6, frame_count/2)
        assert max_frame >= 1, 'Not enough frames to sample from.'
        return frame_index

    def rand_pixel_index(self):
        """
        Function to generate a randomStuff starting pixel for cropping the height and width of the frames.
        :return: 2 ints representing the starting pixel's x and y coordinates.
        """
        # if the frame is precropped, then 50 pixels are removed from the right and left each.
        if self.precrop:
            width = self.width - 100
        else:
            width = self.width
        height_index = np.random.randint(0, self.height - self.resize_height)
        width_index = np.random.randint(0, width - self.resize_width)

        return height_index, width_index

    def load_frames(self, vid_path, frame_index, pixel_index):
        """
        Function to load the video frames for the sample.
        :param vid_path: (str) The path at which the sample video is located.
        :param frame_index: (int) The starting frame index for the sample.
        :param pixel_index: (tuple: int, int) The height and width indices of the pixel at which to crop.
        :return: np array representing the sample video clip.
        """
        buffer = np.empty((self.clip_len, self.resize_height, self.resize_width, self.channels), np.dtype('float32'))

        # retrieve and crop each frame for the clip
        for i in range(self.clip_len):
            # retrieve the frame (next 9 lines)
            frame_num = frame_index + (i * self.skip_len)
            frame_name = NTUDataset.make_frame_name(frame_num=frame_num)
            frame_path = os.path.join(vid_path, frame_name)
            assert os.path.exists(frame_path), 'Frame path {} DNE.'.format(frame_path)
            try:
                frame = cv2.imread(frame_path)
            except:
                print('The image did not successfully load.')
            frame = np.array(frame).astype(np.float32)

            # crop the frame (next 3 lines)
            #height_index, width_index = pixel_index
            #frame = self.crop_frame(frame=frame, h_index=height_index, w_index=width_index)
            
            # normalize frame 
            frame = NTUDataset.normalize_frame(frame)

            # add the frame to the buffer (clip)
            buffer[i] = frame

        return buffer

    @staticmethod
    def make_frame_name(frame_num):
        """
        Function to correctly generate the correctly formatted .jpg file name for the frame.
        :param frame_num: The frame number captured in the file.
        :return: str representing the file name.
        """
        #return str(frame_num).zfill(3) + '.jpg'
        return str(frame_num).zfill(3) + '.png'

    def crop_frame(self, frame, h_index, w_index):
        """
        Function that crops a frame.
        :param frame: (array-like) The frame to crop.
        :param h_index: (int) The height index to start the crop.
        :param w_index: (int) The width index to start the crop.
        :return: np array representing the cropped frame
        """
        frame = np.array(frame).astype(np.float32)
        if self.precrop:
            frame = frame[:, 50:-50]
        cropped_frame = frame[h_index:h_index + self.resize_height, w_index:w_index + self.resize_width]

        return cropped_frame

    @staticmethod
    def normalize_clip_local(tensor, mean, std):
        #mean=self.mean
        #std=self.std

        rep_mean = mean * (tensor.size()[0] // len(mean))
        rep_std = std * (tensor.size()[0] // len(std))

        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)
            
        return tensor

        
    @staticmethod
    def normalize_frame(frame):
        """
        Function to normalize the pixel values in the frame to be between 0 and 1.
        :param frame: (array-like) The frame to be normalized.
        :return: (np array) The normalized frame.
        """
        frame = np.array(frame).astype(np.float32)
        return np.divide(frame, 255.0)

    @staticmethod
    def to_tensor(sample):
        """
        Function to convert the sample clip to a tensor.
        :param sample: (np array) The sample to convert.
        :return: a tensor representing the sample clip.
        """
        # np is (temporal, height, width, channels)
        # pytorch tensor is (channels, temporal, height, width)
        sample = np.transpose(sample, (3, 0, 1, 2))
        tensor = torch.from_numpy(sample)
        return tensor
    
if __name__ == '__main__':
#     data_root_dir, train_split, test_split, dict_file = ucf101_config()

    HEIGHT = 64
    WIDTH = 64
    device = 'cpu'
    FRAMES = 32
    SKIP_LEN = 1
    BATCH_SIZE = 3
    data_root_dir = '/home/c3-0/yogesh/data/ntu60/frames64x64_poseCrop/'
    train_split = 'train16.txt'
    dict_file = 'mapping_train.pickle'
        
    # data
    trainset = NTUDataset(root_dir=data_root_dir, data_file=train_split, dict_file=dict_file, clip_len=FRAMES, skip_len=SKIP_LEN)
    trainset.__getitem__(0)
    #trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    #testset = UCF101Dataset(root_dir=data_root_dir, data_file=test_split, param_file=param_file,
    #                        resize_height=HEIGHT, resize_width=WIDTH,
    #                        clip_len=FRAMES, skip_len=SKIP_LEN)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    for batch_idx, (caption,vid1,vid2,frame_index, frame_count,action) in enumerate(trainloader):
        print(frame_index.type())
        print(frame_count)
        print(caption[0])
        print(caption[1])
        print(vid1.size())
        print(vid2.size())
        print(batch_idx)
        print(action)
        print((action.size()))
        break

