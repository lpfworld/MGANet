# -*- coding:utf-8 -*-
"""
@author:lpf
@file: make_dataset.py
@time: 2020/06/28
通过CSRNet来生成.h5的真是真是密度图
"""
import scipy.io as io
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy
from matplotlib import cm as CM
from image import *

def gaussian_filter_density(gt):
    print(gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    print(gt_count)
    if gt_count == 0:
        return density

    # pts = np.array(zip(np.nonzero(gt)[1], np.nonzero(gt)[0]))作者写错了
    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048

    # build kdtree 寻找最临近点 #构造KDTree寻找相邻的人头位置
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
            # 相邻三个人头的平均距离，其中beta=0.3
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print('done.')
    return density

# 这里有几个地方需要解释一下：
# 1、构造KDTree调用的是scipy包中封装好的函数，其中leafsize表示的是最大叶子数，
# 如果图片中人头数量过多，可以自行修改其数值。
# 2、tree.query（）中参数k=4查询的是与当前结点相邻三个结点的位置信息，
# 因为distances[i][0]表示的是当前结点。
# 3、在论文中beta=0.3，因为这里计算的是三个点的平均距离，
# 所以除以3然后乘以beta相当于直接乘以0.1。

root = '/home/lpf/PycharmProjects/CrowdCountingDataSets/ShanghaiTech/ShanghaiTechDataSet_csrnet/'

part_A_train = os.path.join(root,'part_A_final/train_data/','images')
part_A_test = os.path.join(root,'part_A_final/test_data/','images')
part_B_train = os.path.join(root,'part_B_final/train_data/','images')
part_B_test = os.path.join(root,'part_B_final/test_data/','images')

#一，针对part_A_final 产生.h5真值图
path_sets = [part_A_train,part_A_test]
img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)
print(len(img_paths))

for img_path in img_paths:
    print(img_path)
    mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_'))
    img= plt.imread(img_path)
    k = np.zeros((img.shape[0],img.shape[1]))
# 生成密度图时首先生成一个和原始图像大小相同的全0矩阵(np.zeros)

    gt = mat["image_info"][0,0][0,0][0]
# 然后遍历标注文件中每一个位置坐标

    for i in range(0,len(gt)):
        if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
            k[int(gt[i][1]),int(gt[i][0])]=1
# 将矩阵中对应的点置为1

    k = gaussian_filter_density(k)
    with h5py.File(img_path.replace('.jpg','.h5').replace('images','ground_truth'), 'w') as hf:
            hf['density'] = k
# 最后调用高斯核函数生成密度图并保存成h5py格式的文件
plt.imshow(Image.open(img_paths[0]))

gt_file = h5py.File(img_paths[0].replace('.jpg','.h5').replace('images','ground_truth'),'r')
groundtruth = np.asarray(gt_file['density'])
plt.imshow(groundtruth,cmap=CM.jet)

np.sum(groundtruth)
# don't mind this slight variation

#二，now generate the ShanghaiB's ground truth
# path_sets = [part_B_train,part_B_test]
# img_paths = []
# for path in path_sets:
#     for img_path in glob.glob(os.path.join(path, '*.jpg')):
#         img_paths.append(img_path)
#
# for img_path in img_paths:
#     print (img_path)
#     mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_'))
#     img= plt.imread(img_path)
#     k = np.zeros((img.shape[0],img.shape[1]))
#     gt = mat["image_info"][0,0][0,0][0]
#     for i in range(0,len(gt)):
#         if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
#             k[int(gt[i][1]),int(gt[i][0])]=1
#     k = gaussian_filter(k,15)
#     with h5py.File(img_path.replace('.jpg','.h5').replace('images','ground_truth'), 'w') as hf:
#             hf['density'] = k
# plt.imshow(Image.open(img_paths[0]))



