import itertools as it
import random
import pickle
import os

data_file = 'train16.txt'
with open(data_file) as f:
    x = f.readlines()
f = [line.strip() for line in x]
print((f[0].split(' ')))

cnt=len(f)
print(cnt)
# for data_file in f:
#     sample_info = data_file.split(' ')
#     sample_id = sample_info[0][0:sample_info[0].index('/')]
#     if(int(sample_id)<50):
#         temp = sample_info[0] + " 1 " + sample_info[1]
#         # print(temp)
#         # break
#         cnt += 1

# print(cnt)

# path='/home/namanb/Desktop/Vidgen_from_text/networks'
# print(os.listdir(path))
# print(len(os.listdir(path)))

def create_mapping(mapping_file="mapping_train2.pickle"):
    """
    This will delete the previuos mapping.pickle and make a new random mapping
    """
    # os.system("rm "+mapping_file)
    mapping = dict()
    # index = 0
    # numclasses = 101
    print(cnt)
    i=0
    for j in range(49):
        idx = int(f[i][0:f[i].index('/')])
        # print(idx) 
        list1 = [] 
        while i<cnt and idx == int(f[i][0:f[i].index('/')]):
            # print(f[i])
            list1.append(f[i])
            i = i + 1
        list2 = random.sample(list1, len(list1))
        random_map = dict(zip(list1,list2))
        mapping[int(idx)] = random_map
        print(len(mapping[idx]))
        print(i)
        # break
    
    
    os.system("touch "+mapping_file)
    pickle_out = open(mapping_file,"wb")
    pickle.dump(mapping, pickle_out)
    print(f[i-1])

# create_mapping()
# pickle_in = open("mapping_train2.pickle",'rb')
# vid_dict = pickle.load(pickle_in)
# print((vid_dict[1]['1/S017P009R002A001 1 98']))
    
# mapping = dict()
# index = 0
# for i in range(1,5):
#     list1 = [] 
#     while f[index].endswith(' '+str(i)):
#         list1.append(f[index])
#         index = index + 1
#     list2 = random.sample(list1, len(list1))
#     random_map = dict(zip(list1,list2))
#     mapping[i] = random_map 
# print((mapping[4]['BabyCrawling/v_BabyCrawling_g13_c06.avi 4']))
# s = mapping[4]['BabyCrawling/v_BabyCrawling_g13_c06.avi 4']
# i = s.index('/')
# print(i)
# print(s[0:i])
# st = s[0:i]
# # i = 0
# wordlist = []
# index = 0
# for j in range(1,i):
#     if s[j].isupper():
#         wordlist.append(st[index:j])
#         index = j
# wordlist.append(st[index:i])
# print(wordlist)
# print([word.lower() for word in wordlist])
# eyemakeup = [""]
# for i in f:
#     if i.endswith(' 1'):
#         eyemakeup.append(i)

# print(eyemakeup[0])
# print(len(eyemakeup))
# list1 = eyemakeup
# list2 = eyemakeup
# f = <|AssociationThread[list1 -> list2], AssociationThread[list2 -> list1]|>;
# # for i in range(0,2):
#     print(f[i])
# print(f[32].endswith('1'))
# a = ['1','2','3','4','5']
# b = random.sample(a, len(a))

# c = dict(zip(a, b))
# print(c)

# a = {1:"6",2:"2",3:"f"}
# b = {1:"2",2:"3",3:"g"}

# dict_out = {'1':a, '2':b}
# pickle_out = open("dict.pickle","wb")
# pickle.dump(dict_out, pickle_out)
# pickle_out.close()
# os.system("rm dict.pickle")
# pickle_in = open("dict.pickle","rb")
# example_dict = pickle.load(pickle_in)
# print(example_dict['1'][1])

# pickle_in = open("mapping_train.pickle",'rb')
# vid_dict = pickle.load(pickle_in)
# print(len(vid_dict[1]))


# def get_action_wordlist(self,action):
#     """
#     this takes the complete action of the video and returns a list
#     containing the words of the action
#     """
#     wordlist = []
#     index = 0
#     l = len(action)
#     for i in range(1,l):
#         if action[i].isupper():
#             wordlist.append(action[index:i])
#             index = i
#     wordlist.append(action[index:l])

#     return [word.lower() for word in wordlist]