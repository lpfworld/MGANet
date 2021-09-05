# -*- coding:utf-8 -*-
"""
@author:lpf
@file: test.py
@time: 2020/06/29
最后，检测此模型在不可视数据上的表现情况。我们将使用val.ipynb文件来验证结果。
请记住将路径更改为预训练权值和图像。
"""
import glob
import torchvision.transforms.functional as F
from image import *
from model import CSRNet
import torch

from torchvision import transforms
transform = transforms.Compose([
    transforms.ToTensor(),transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
                   ])

root = '/home/lab421/lpf/CrowdCounting-ImageCaption/0-CSRNet-pytorch-master/data/ShanghaiTechDataSet_csrnet/'
#now generate the ShanghaiA's ground truth
part_A_train = os.path.join(root,'part_A_final/train_data','images')
part_A_test = os.path.join(root,'part_A_final/test_data','images')
part_B_train = os.path.join(root,'part_B_final/train_data','images')
part_B_test = os.path.join(root,'part_B_final/test_data','images')
path_sets = [part_B_test]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)
model = CSRNet()

model = model.cuda()

checkpoint = torch.load('/home/lab421/lpf/CrowdCounting-ImageCaption/0-CSRNet-pytorch-master/best_model/00903model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
# 检查测试图像上的MAE（平均绝对误差），评估我们的模型：
mae = 0
mse = 0
for i in range(len(img_paths)):
    img = 255.0 * F.to_tensor(Image.open(img_paths[i]).convert('RGB'))

    img[0,:,:]=img[0,:,:]-92.8207477031
    img[1,:,:]=img[1,:,:]-95.2757037428
    img[2,:,:]=img[2,:,:]-104.877445883
    img = img.cuda()
    img = transform(Image.open(img_paths[i]).convert('RGB')).cuda()
    gt_file = h5py.File(img_paths[i].replace('.jpg','.h5').replace('images','ground_truth'),'r')

    groundtruth = np.asarray(gt_file['density'])
    output = model(img.unsqueeze(0))

    mae += abs(output.detach().cpu().sum().numpy()-np.sum(groundtruth))
    mse += (output.detach().cpu().sum().numpy()-np.sum(groundtruth))*(output.detach().cpu().sum().numpy()-np.sum(groundtruth))
    print('第{}张图片的真实值结果是{}，模型的结果是{}'.format(i,output.detach().cpu().sum().numpy(),np.sum(groundtruth)))

mae= mae/len(img_paths)
mse= np.sqrt(mse/len(img_paths))

print('最终MAE结果是{}，最终MSE结果是{}'.format(mae,mse))

