import os

def get_no_of_frames(self,path):
    """
    returns the number of frames of the given video path"
    """
    return len(os.listdir(path))

def get_vid_paths(self, action, sample_id):
    """
    Function to get the paths at which the two sample views are located.
    :param action: (int) The action class that the sample captures.
    :param sample_id: (str) The id for the sample from the data file.
    :return: 2 strings representing the paths for the sample views.
    """
    vid_path = os.path.join(root_dir, str(action), sample_id)
    return vid_path

def process_index(self, index):
    """
    Function to process the information that the data file contains about the sample.
    The line of information contains the sample name as well as the number of available frames from each view
    :param index: (int) The index of the sample.
    :return: the action class, sample_id, two floats representing the viewpoint angles, and two ints representing
                the number of frames in view1 and view2
    """
    sample_info = self.data_file[index].split(' ')
    sample_action = sample_info[0][0:sample_info[0].index('/')]
    sample_id = sample_info[0][sample_info[0].index('/') + 1:]
    sample_action_no = sample_info[1]

    info = (sample_info, sample_action, sample_id, sample_action_no)
    return info


# path="testlist0.txt"
# root_dir = "/data/namanb/resized-64x64/"
listfile = "caption.txt"
data_file = 'train16.list'

# if(not (os.path.isfile(listfile))):
    # os.system("touch "+listfile)

# with open(data_file) as f:
#     x = f.readlines()
# f = [line.strip() for line in x]

# cnt=0
# for data_file in f:
#     sample_info = data_file.split(' ')
#     sample_id = sample_info[0][0:sample_info[0].index('/')]
#     if(int(sample_id)<50):
#         cnt += 1

# print(cnt*3)

trainlist = []
# for data_file in f:
#     sample_info = data_file.split(' ')
#     # print(info[0][:-4])
#     # dirpath = root_dir + info[0][:-4]
#     # if(os.path.isdir(dirpath)):
#         # print(dirpath)
#     sample_id = sample_info[0][0:sample_info[0].index('/')]
#     if(int(sample_id)<50):
#         temp = sample_info[0] + " 1 " + sample_info[1]
#         trainlist.append(temp)
#         # temp = sample_info[0] + " 2 " + sample_info[2]
#         # trainlist.append(temp)
#         # temp = sample_info[0] + " 3 " + sample_info[3]
#         # trainlist.append(temp)

#     # print(sample_id)
    # print(path)

trainlist.append("drink water")
trainlist.append("eat meal")
trainlist.append("brush teeth")
trainlist.append("brush hair")
trainlist.append("drop")
trainlist.append("pick up")
trainlist.append("throw")
trainlist.append("sit down")
trainlist.append("stand up")
trainlist.append("clapping")
trainlist.append("reading")
trainlist.append("writing")
trainlist.append("tear up paper")
trainlist.append("put on jacket")
trainlist.append("take off jacket")
trainlist.append("put on a shoe")
trainlist.append("take off a shoe")
trainlist.append("put on glasses")
trainlist.append("take off glasses")
trainlist.append("put on a hat cap")
trainlist.append("take off a hat cap")
trainlist.append("cheer up")
trainlist.append("hand waving")
trainlist.append("kicking something")
trainlist.append("reach into pocket	")
trainlist.append("hopping")
trainlist.append("jump up")
trainlist.append("phone call")
trainlist.append("play with phone tablet")
trainlist.append("type on a keyboard")
trainlist.append("point to something")
trainlist.append("taking a selfie")
trainlist.append("check time from watch")
trainlist.append("rub two hands")
trainlist.append("nod head bow")
trainlist.append("shake head")
trainlist.append("wipe face")
trainlist.append("salute")
trainlist.append("put palms together")
trainlist.append("cross hands in front")
trainlist.append("sneeze cough")
trainlist.append("staggering")
trainlist.append("falling down")
trainlist.append("headache")
trainlist.append("chest pain")
trainlist.append("back pain")
trainlist.append("neck pain")
trainlist.append("nausea vomiting")
trainlist.append("fan self")

print(len(trainlist))

trainlist = [word+"\n" for word in trainlist]
trainlist[-1]=trainlist[-1][:-1]

my_file = open(listfile,'w')
my_file.writelines(trainlist)
my_file.close()

