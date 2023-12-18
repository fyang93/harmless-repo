import time
import requests
import json

import os
import sys
import yaml
import random
import argparse
import logging
import numpy as np

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from datetime import datetime, timezone, timedelta
"""
BeiJing = timezone(timedelta(hours=8))

def timestr():
    UTC = datetime.utcnow().replace(tzinfo=timezone.utc)
    return UTC.astimezone(BeiJing).strftime('%m-%d %H:%M:%S')

ROOT = os.path.dirname(os.path.abspath(__file__))

class Logger(object):
    level_relations = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }
    def __init__(self, filename, level='info', mode="a", fmt='%(asctime)s - %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S"):
        format_str = logging.Formatter(fmt, datefmt)
        self.logger = logging.getLogger(filename)
        self.logger.setLevel(self.level_relations.get(level))
        console = logging.StreamHandler()
        console.setFormatter(format_str)
        fh = logging.FileHandler(filename=filename, mode=mode, encoding='utf-8')
        fh.setFormatter(format_str)
        self.logger.addHandler(console)
        self.logger.addHandler(fh)
"""

def parse_args(prefix):
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", required=True, type=str, help="File suffix")
    # Basic
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--root", default='./', type=str)
    parser.add_argument("--folder", default='result', type=str)
    parser.add_argument("--log", choices=["debug", "info"], default="debug", help='Log print level.')
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    # Model
    parser.add_argument("-m", "--model", default="resnet18", choices=["resnet18", "resnet34", "resnet50"], type=str, help="Backbone Network {Res-18, Res-34, Res-50}.")
    parser.add_argument("--fine_tune_path", default=None, type=str)
    parser.add_argument("--num_class", default=10, type=int, help="Number of image classes")
    # Dataset
    parser.add_argument("-d", "--dataset", default="CIFAR10", type=str, help="Dataset.")   # "CIFAR10"
    parser.add_argument("--normalize", action="store_true", default=False)
    #parser.add_argument("--data_path", default="./data/CIFAR10/", type=str, help="Dataset.")
    # Optimizer & scheduler
    parser.add_argument("--optimizer", default="SGD", type=str)
    parser.add_argument("--lr", default=0.1, type=float, help="Learning rate")
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=5e-4, type=float)
    # parser.add_argument("--gamma", default=0.1, type=float)
    # parser.add_argument("--milestones", default=[50, 60], type=list)
    # Hyper Parameter
    parser.add_argument("--epoch_num", default=60, type=int, help="Number of Epochs")
    parser.add_argument("--start_epoch", default=1, type=int, help="Train start from # epochs")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size")

    parser.add_argument("--lam", default=1.0, type=float)

    args = parser.parse_args(sys.argv[1:])

    args.result_path = os.path.join(args.root, args.folder, f"{prefix}_{args.suffix}")
    if args.fine_tune_path:
        args.result_path = args.result_path + "_finetune"
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    if args.dataset == "imagenet" or args.dataset == "tiny-imagenet-200":
        args.img_size = 224
    elif args.dataset == "CIFAR10":
        args.img_size = 32
    args.data_path = os.path.join(args.root, "data/CIFAR10/")

    return args


def setup_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        cudnn.deterministic = True

def logits_accuracy(logits, labels):
    preds = logits.argmax(dim=-1)
    acc = (preds==labels).sum().item()/labels.size(0)
    return acc

"""
@torch.enable_grad()
def PGD(model, images, labels, steps, step_size, epsilon):
    model.eval()
    detach_images = images.detach()
    images_adv = detach_images + 0.001 * torch.randn_like(detach_images)
    images_adv = torch.clamp(images_adv, min=0, max=1)

    for _ in range(steps):
        images_adv.requires_grad_(True)
        logits = model(images_adv)
        loss = F.cross_entropy(logits, labels)
        grad = torch.autograd.grad(loss, [images_adv])[0].detach()

        images_adv = images_adv.detach() + step_size * torch.sign(grad)
        images_adv = torch.minimum(torch.maximum(images_adv, images - epsilon), images + epsilon)
        images_adv = torch.clamp(images_adv, 0.0, 1.0)

    return images_adv

def pgd10(model, images, labels):
    return PGD(model, images, labels, steps=10, step_size=2/255, epsilon=8/255)

def pgd20(model, images, labels):
    return PGD(model, images, labels, steps=20, step_size=1/255, epsilon=8/255)

@torch.enable_grad()
def PGDX(model, images, labels, steps, step_size, epsilon):
    model.eval()
    with torch.no_grad():
        alabels = model(images).detach().argmax(dim=-1)
    detach_images = images.detach()
    images_adv = detach_images + 0.001 * torch.randn_like(detach_images)
    images_adv = torch.clamp(images_adv, min=0, max=1)

    for _ in range(steps):
        images_adv.requires_grad_(True)
        logits = model(images_adv)
        loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits, alabels)
        grad = torch.autograd.grad(loss, [images_adv])[0].detach()

        images_adv = images_adv.detach() + step_size * torch.sign(grad)
        images_adv = torch.minimum(torch.maximum(images_adv, images - epsilon), images + epsilon)
        images_adv = torch.clamp(images_adv, 0.0, 1.0)

    return images_adv

def pgdx10(model, images, labels):
    return PGDX(model, images, labels, steps=10, step_size=2/255, epsilon=8/255)


@torch.enable_grad()
def PGDR(model, images, labels, steps, step_size, epsilon):
    model.eval()
    images_adv = images.detach()


    for _ in range(steps):
        images_adv = torch.clamp(images_adv + 0.001 * torch.randn_like(images_adv), min=0, max=1)
        images_adv.requires_grad_(True)
        loss = F.cross_entropy(model(images_adv), labels)
        grad = torch.autograd.grad(loss, [images_adv])[0].detach()

        images_adv = images_adv.detach() + step_size * torch.sign(grad)
        images_adv = torch.minimum(torch.maximum(images_adv, images - epsilon), images + epsilon)
        images_adv = torch.clamp(images_adv, 0.0, 1.0)

    return images_adv

def pgdr10(model, images, labels):
    return PGDR(model, images, labels, steps=10, step_size=2/255, epsilon=8/255)

class WeChat:
    def __init__(self, CORPID, CORPSECRET):
        self.CORPID = CORPID  #企业ID，在管理后台获取
        self.CORPSECRET = CORPSECRET#自建应用的Secret，每个自建应用里都有单独的secret
        self.AGENTID = '1000002'  #应用ID，在后台应用中获取
        self.TOUSER = "ChenRenJie"  # 接收者用户名,多个用户用 |分割，@all表示全体用户
        self.TOPARY = "2"    #部门ID
        self.CONF = "access_token.conf"

    def _get_access_token(self):
        url = 'https://qyapi.weixin.qq.com/cgi-bin/gettoken'
        params = {
            'corpid': self.CORPID,
            'corpsecret': self.CORPSECRET,
        }
        r = requests.post(url, params=params)
        return r.json()["access_token"]

    def get_access_token(self):
        cur_time = time.time()
        try:
            with open(self.CONF, "r") as f:
                t, access_token = f.read().split()
        except:
            with open(self.CONF, "w") as f:
                access_token = self._get_access_token()
                f.write('\t'.join([str(cur_time), access_token]))
                return access_token
        else:
            if cur_time - float(t) < 7260:
                return access_token
            else:
                with open(self.CONF, 'w') as f:
                    access_token = self._get_access_token()
                    f.write('\t'.join([str(cur_time), access_token]))
                    return access_token

    def send_message(self, payload):
        url = "https://qyapi.weixin.qq.com/cgi-bin/message/send"
        params = {
            "access_token": self.get_access_token()
        }
        r = requests.post(url, params=params, json=payload)
        return r.json()["errmsg"]

    def send_markdown(self, content):
        data = {
            "msgtype": "markdown",
            "touser": self.TOUSER,
            "agentid": self.AGENTID,
            "markdown": {
                "content": content
            },
            "enable_duplicate_check": 0,
            "duplicate_check_interval": 1800
        }
        self.send_message(data)
"""

