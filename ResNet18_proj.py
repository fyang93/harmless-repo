import torch.nn.functional as F
from tqdm.notebook import tqdm
import os
import argparse
import numpy as np
from sympy import *
from sympy.polys.matrices import DomainMatrix
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from pgd10.PGDAT.model import resnet18
from pgd10.PGDAT.attack import PGD20, PGD20_features

from ortho_perturbation_Resnet import HarmlessPerturbation, HarmlessPerturbationLinearLayer
from utils.seed import setup_seed
#from utils.plot import plot_features

EPS = 1e-5

class ForwardFunction(nn.Module):
    def __init__(self, model: nn.Module,):
        super(ForwardFunction, self).__init__()
        self.model = model
        self.conv = model.layer2.block0.conv1  # conv layer to be analyzed

    def _get_features(self, input):
        with torch.no_grad():
            features = self.model._get_x0(input)
        return features

    def forward(self, features):
        x = self.model(features)

        return x


class ForwardFunctionLinearLayer(nn.Module):
    def __init__(self, model: nn.Module,):
        super(ForwardFunctionLinearLayer, self).__init__()
        self.model = model
        self.fc = model.fc  # fully-connected layer to be analyzed

    def _get_features(self, input):
        with torch.no_grad():
            features = self.model._get_fc(input)
        return features

    def forward(self, features):
        x = self.model(features)

        return x


def get_perturbation_shape(model, test_loader):
    model.eval()
    for i, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        if args.input == 'image':
            return tuple(images[0].shape)
        elif args.input == 'feature':
            return tuple(model._get_features(images)[0].shape)




def get_perturbation_projection(projMatrix, delta, mode="parallel", device="cuda:0"):
    # parallel perturbation
    if mode == "parallel":
        para_delta = torch.matmul(projMatrix.to(device), delta.squeeze(dim=2).T.to(device)).T.unsqueeze(dim=2)
        return para_delta

    # orthogonal perturbation
    elif mode == "orthogonal":
        identityMatrix = torch.eye(projMatrix.shape[0], projMatrix.shape[1]).to(device)
        ortho_delta = torch.matmul(identityMatrix - projMatrix.to(device), delta.squeeze(dim=2).T.to(device)).T.unsqueeze(dim=2)
        return ortho_delta


def multiply_perturbation(perturbation, alpha):
    perturbation_reshape = perturbation.resize(features.shape[0], features.shape[1] * features.shape[2] * features.shape[3])
    multiplied_perturbation = torch.zeros_like(perturbation_reshape).to(device)
    for idx in range(features.shape[0]):  # 后期加速 --> 用batch
        multiplied_perturbation[idx] = alpha * perturbation_reshape[idx]
    multiplied_perturbation = multiplied_perturbation.reshape(features.shape)
    return multiplied_perturbation



# random noise
def RANDOM_features(images, input='feature', epsilon=8/255, norm="l_inf"):
    if input == 'feature':
        features_max = torch.max(images.reshape(images.shape[0], -1), 1)[0]
        epsilon = epsilon * features_max
        epsilon = epsilon.unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1) * torch.ones_like(images)

    delta_random = 0.1 * torch.randn(images.shape).to(device)  ## 这里的系数0.1根据具体的任务选取合适的值（如果需要测量随机扰动的acc时）
    delta_random = torch.clamp(delta_random, min=-epsilon, max=epsilon)
    if input == 'feature':
        image_random = delta_random + images
    if input == 'image':
        image_random = torch.clamp(delta_random + images, min=0, max=1)

    return image_random


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def get_angle_between(v1, v2, isdegree=None):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            # >>> angle_between((1, 0, 0), (0, 1, 0))
            # 1.5707963267948966
            # >>> angle_between((1, 0, 0), (1, 0, 0))
            # 0.0
            # >>> angle_between((1, 0, 0), (-1, 0, 0))
            # 3.141592653589793
    """
    if torch.is_tensor(v1):
        v1 = v1.detach().cpu().numpy()
    if torch.is_tensor(v2):
        v2 = v2.detach().cpu().numpy()

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    # radian to degree
    if isdegree:
      angle = np.rad2deg(angle)
    return angle


def init_path(args):
    # model path
    args.model_path = os.path.join(args.save_path, args.model, args.model_path)
    if not os.path.exists(os.path.join(args.save_path, args.model)):
        os.makedirs(os.path.join(args.save_path, args.model))
    print(os.path.abspath(args.model_path))

    # null space path
    #args.nSpace_path = './output/nSpace_ResNet18.npy'
    args.nSpace_path = os.path.join(args.save_path, args.model, 'fc'+args.harmless_space_path)
    print(os.path.abspath(args.nSpace_path))

    # projection matrix path
    #args.projMatrix_path = './output/projMatrix_ResNet18.npy'
    args.projMatrix_path = os.path.join(args.save_path, args.model, args.projection_matrix_path)
    print(os.path.abspath(args.projMatrix_path))


def test(model, test_loader):
    model.eval()
    test_loss_value, test_correct_value = 0, 0
    test_total_num = 0

    for i, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            output = model(images)

        loss = criterion(output, labels)
        test_loss_value += loss.item() * images.size(0)
        test_correct_value += (output.max(1)[1] == labels).sum().item()
        test_total_num += images.size(0)

    avg_loss = test_loss_value / test_total_num
    avg_acc = test_correct_value / test_total_num
    print(f"test_loss: {avg_loss:.4f} test_acc: {avg_acc:.4f}\n")

    return avg_loss, avg_acc



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Orthogonal perturbations on CIFAR10')
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--model', type=str, default='ResNet18')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--test_acc', type=bool, default=False, help="whether to test the classification accuracy of the model")

    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--device2', type=str, default='cuda:4')
    parser.add_argument('--seed', type=str, default=0)
    parser.add_argument('--save_path', type=str, default='./output')
    parser.add_argument('--data_path', type=str, default='~/data/cifar10')
    parser.add_argument('--model_path', type=str, default='ResNet18.pth') #"./exp/result0901/PGDAT_ST_pgd20/checkpoint.pth", "./exp/result0901/PGDAT_pgd10_pgd20/checkpoint.pth"
    parser.add_argument('--harmless_space_path', type=str, default='nSpace.npy')
    parser.add_argument('--projection_matrix_path', type=str, default='projMatrix.npy')

    # calculate harmless space
    parser.add_argument('--test_harmlessness', type=bool, default=False, help="whether to test the harmlessness of generated perturbation")
    parser.add_argument('--input', type=str, default="feature", help=" 'image' or 'feature' ")
    parser.add_argument('--layer', type=str, default="convolutional", help="'fully-connected' or 'convolutional' layer")
    # if choose convolutional layer
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--padding', type=int, default=1)

    # add noise
    parser.add_argument('--perturbation_mode', type=str, default="parallel", help="'parallel' or 'orthogonal' ")
    parser.add_argument('--alpha', type=float, default=1, help="the magnitude of perturbation")
    parser.add_argument('--epsilon', type=float, default=8/255)
    parser.add_argument('--attack_method', type=str, default="pgd20", help=" 'pgd20' or 'random' ")

    args = parser.parse_args()

    setup_seed(seed=args.seed)
    init_path(args)
    device = args.device
    device2 = args.device2

    # load data
    if args.dataset == 'CIFAR10':
        train_set = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transforms.ToTensor())
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
        test_set = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transforms.ToTensor())
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # load model
    criterion = nn.CrossEntropyLoss()
    if args.model == 'ResNet18':
        model = resnet18(num_classes=10).to(device)
    #model_state_dict = torch.load(args.model_path, map_location=device)
    model_state_dict = torch.load(args.model_path, map_location=device)["model"]
    model.load_state_dict(model_state_dict)
    print(model)

    if args.test_acc:
        test(model, test_loader)

    # modify input of the model
    if args.input == 'feature':
        if args.layer == 'convolutional':
            model = ForwardFunction(model=model)
        elif args.layer == 'fully-connected':
            model = ForwardFunctionLinearLayer(model=model)
        model = model.to(device)
    perturbation_shape = get_perturbation_shape(model, test_loader)

    # get harmless subspace
    if os.path.exists(args.nSpace_path):
        nSpace = np.load(args.nSpace_path)
        nSpace = torch.tensor(nSpace).float().to(device)  # [65536, 32768]
    else:
        # calculate harmless subspace
        if args.layer == 'convolutional':
            stride = args.stride
            padding = args.padding
            conv = model.conv
            calculator = HarmlessPerturbation(kernel=conv, stride=stride, padding=padding,
                                                perturbation_shape=perturbation_shape, device=device,
                                                nSpace_path=args.nSpace_path)
            delta, orthogonal_bases = calculator.get_delta()

        elif args.layer == 'fully-connected':   ## 全连接层需要修改
            weight = model.weight  ## 需要定义哪一层weight
            calculator = HarmlessPerturbationLinearLayer(weight=weight.T, device=device, nSpace_path=args.nSpace_path)
            delta, orthogonal_bases = calculator.get_delta()
            # dM = DomainMatrix.from_Matrix(Matrix(weight.cpu().detach().numpy()))
            # r = dM.to_field().nullspace().transpose().to_Matrix()
            # r = np.array(r).astype(np.float64)
            # nSpace = torch.Tensor(r)

        # load harmless subspace
        nSpace = np.load(args.nSpace_path)
        nSpace = torch.tensor(nSpace).float().to(device)

    # whether to test the harmlessness of generated perturbation
    if args.test_harmlessness:
        if args.layer == 'convolutional':
            conv = model.conv.double()
            image = torch.randn(perturbation_shape).unsqueeze(dim=0).double().to(device)
            print(nSpace.shape[1])

            for i in range(nSpace.shape[1]):
                print(i)
                delta = nSpace[:, i].reshape(perturbation_shape).unsqueeze(dim=0).double()
                print("isHarmless:", (((conv(image+delta) - conv(image)) < EPS)+0).sum())
                print("distance:", torch.norm((conv(image+delta) - conv(image)).reshape(-1), p=2))
                #print("output (a kernel):", conv(image+delta) - conv(image))
        elif args.layer == 'fully-connected':
            weight = model.weight.double()  ## 需要定义哪一层weight
            ## 需要补充全连接层的测试结果 !!!!!
            image = torch.randn(perturbation_shape).unsqueeze(dim=0).double().to(device)
            print(nSpace.shape[1])

            for i in range(nSpace.shape[1]):
                print(i)
                delta = nSpace[:, i].reshape(perturbation_shape).unsqueeze(dim=0).double()
                print("isHarmless:", (((weight(image + delta) - weight(image)) < EPS) + 0).sum())
                print("distance:", torch.norm((weight(image + delta) - weight(image)).reshape(-1), p=2))


    exit(0)


    # test the impact of harmless and harmful perturbations
    ori_num, proj_num, ptb_num, total_num = 0, 0, 0, 0
    total_dist, total_dist_ptb = 0, 0  # distance
    DIST, DIST_PTB = [], []  # distance
    proj_angle = []  # projection angle
    # attack parameter
    epsilon = args.epsilon
    attack_method = args.attack_method

    # get projection matrix P
    if os.path.exists(args.projMatrix_path):
        projMatrix = np.load(args.projMatrix_path)  # [65536, 65536]
        projMatrix = torch.tensor(projMatrix).float().to(device)
    else:
        projMatrix = torch.matmul(torch.matmul(nSpace, torch.inverse(torch.matmul(nSpace.T, nSpace))), nSpace.T)  # 这行命令需要 49GB 显存, 当投影矩阵为 [65536, 65536]
        np.save(args.projMatrix_path, np.array(projMatrix.detach().cpu().numpy()))
    #print("projMatrix", projMatrix.shape, projMatrix)

    setup_seed(seed=args.seed)
    for i, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        model = model.float().to(device)
        if args.input == 'feature':
            features = model._get_features(images)
            #plot_features(features.detach().cpu().numpy())
        elif args.input == 'image':
            features = images

        # original perturbation
        if attack_method == "pgd20":  # pgd attack
            features_perturb = PGD20_features(model, features, labels, input=args.input, epsilon=args.epsilon) #PGD20(model, features_norm, labels)
        elif attack_method == "random": # random noise
            features_perturb = RANDOM_features(features, input=args.input, epsilon=args.epsilon, norm="l_inf")
        ori_delta = features_perturb - features
        #print("delta", ori_delta.max(), ori_delta.min(), ori_delta)  # features: > 0.0314    images: 0.0314

        ori_delta_reshape = ori_delta.resize(features.shape[0], features.shape[1]*features.shape[2]*features.shape[3], 1)

        # projection perturbation
        if args.perturbation_mode == "parallel":
            para_delta = get_perturbation_projection(projMatrix, ori_delta_reshape, mode="parallel", device=device2)
            proj_delta = para_delta.reshape(features.shape)
        elif args.perturbation_mode == "orthogonal":
            ortho_delta = get_perturbation_projection(projMatrix, ori_delta_reshape, mode="orthogonal", device=device2)
            proj_delta = ortho_delta.reshape(features.shape)

        # whether to increase the magnitude of perturbation, proj_delta = alpha * proj_delta
        if args.alpha != 1:
            print(f"increase the magnitude of perturbation, alpha={args.alpha}")
            ori_proj_delta = proj_delta.to(device)
            proj_delta = multiply_perturbation(perturbation=proj_delta, alpha=args.alpha)


        # projection image
        features_proj = features + proj_delta.to(device)
        if args.input == 'image':
            ## if image, then the parallel perturbation should not reach the bound [0,1]
            features_proj = torch.clamp(features_proj, min=0, max=1)  ## 需要修改，不能用超过bound然后截断，应该限制扰动的大小使其不达到bound !!!!


        for img_ori, img_proj, img_ptb, lbl in zip(features, features_proj, features_perturb, labels):
            img_ori, img_proj = img_ori.unsqueeze(dim=0), img_proj.unsqueeze(dim=0)
            img_ptb = img_ptb.unsqueeze(dim=0)

            # improve precision
            # img_ori = img_ori.double().to(device2)
            # img_proj = img_proj.double().to(device2)
            # img_ptb = img_ptb.double().to(device2)
            # model = model.double().to(device2)

            y_ori = model(img_ori)
            y_proj = model(img_proj)
            y_ptb = model(img_ptb)
            # print(y_ori)
            # print(y_proj)

            # check whether attack is successful
            if args.perturbation_mode == "parallel":
                distance = torch.norm(y_proj - y_ori, p=2, dim=1)  # parallel / harmless perturbation
                DIST.append(distance)
                total_dist += distance.sum()
                # check whether the distance is not too large
                if ((distance < EPS) + 0).sum() < img_ori.size(0):
                    print("Large output distance (para.):", distance.detach().cpu().numpy())

            elif args.perturbation_mode == "orthogonal":
                distance_ptb = torch.norm(y_proj - y_ptb, p=2, dim=1) # orthogonal / harm perturbation
                DIST_PTB.append(distance_ptb)
                total_dist_ptb += distance_ptb.sum()
                # check whether the distance is not too large
                if ((distance_ptb < EPS) + 0).sum() < img_ori.size(0):
                    print("Large output distance (ortho.):", distance_ptb.detach().cpu().numpy())

            total_num += img_ori.size(0)
            ori_num += int(y_ori.max(1)[1].cpu().numpy() == lbl.cpu().numpy())
            proj_num += int(y_proj.max(1)[1].cpu().numpy() == lbl.cpu().numpy())
            ptb_num += int(y_ptb.max(1)[1].cpu().numpy() == lbl.cpu().numpy())



    if args.perturbation_mode == "parallel":   # parallel perturbation
        print("ori_num: {}, proj_num: {}, total_num: {}".format(ori_num, proj_num, total_num))
        print("proj_acc: {}, ori_acc: {}".format(proj_num / total_num, ori_num / total_num))
        print("total_dist: {}, avg_dist: {}".format(total_dist, total_dist / total_num))
        DIST = torch.stack(DIST).reshape(-1).detach().cpu().numpy()

    elif args.perturbation_mode == "orthogonal":   # orthogonal perturbation
        print("\nptb_num: {}, proj_num: {}, total_num: {}".format(ptb_num, proj_num, total_num))
        print("ptb_acc: {}, proj_acc: {}".format(ptb_num / total_num, proj_num / total_num))
        print("ptb_total_dist: {}, avg_dist: {}".format(total_dist_ptb, total_dist_ptb / total_num))
        DIST_PTB = torch.stack(DIST_PTB).reshape(-1).detach().cpu().numpy()

    exit(0)








    # 12.12 计算扰动的角度
    # # calculate angle between original delta & parallel delta
    # for idx in range(features.shape[0]):  # 后期加速 --> 用batch
    #     ori_delta_idx = ori_delta.reshape(features.shape[0], -1)[idx] # proj_delta.reshape(features.shape[0], -1)[idx]
    #     para_delta_idx = para_delta.reshape(features.shape[0], -1)[idx]
    #
    #     proj_angle_idx = get_angle_between(ori_delta_idx, para_delta_idx, isdegree=True)
    #     proj_angle.append(proj_angle_idx)

    # ##### 注意这段结束的 proj_delta = alpha * proj_delta, 若不需要则把这整段注释掉
    # # proj_delta = alpha * ortho perturbation
    # alpha = 1.0
    # proj_delta_reshape = proj_delta.resize(features.shape[0], features.shape[1]*features.shape[2]*features.shape[3])
    # max_ortho_delta = torch.zeros_like(proj_delta_reshape).to(device)
    # for idx in range(features.shape[0]):  # 后期加速 --> 用batch
    #     max_ortho_delta[idx] = alpha * proj_delta_reshape[idx]
    #     list = [i * batch_size + idx, \
    #             torch.norm(max_ortho_delta[idx], p=np.inf).detach().cpu().numpy(), \
    #             torch.norm(max_ortho_delta[idx], 2).detach().cpu().numpy(), \
    #             torch.norm(proj_delta_reshape[idx], p=np.inf).detach().cpu().numpy(), \
    #             torch.norm(proj_delta_reshape[idx], 2).detach().cpu().numpy()]
    #     data = pd.DataFrame([list])
    #     data.to_csv(f"./output/ortho_attack_norm_{attack_method}/alpha{alpha}.csv", mode='a', header=False, index=False)
    # max_ortho_delta = max_ortho_delta.reshape(features.shape)
    # proj_delta = max_ortho_delta
    #