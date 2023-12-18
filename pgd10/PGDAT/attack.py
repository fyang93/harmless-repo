import torch
import torch.nn.functional as F


@torch.enable_grad()
def pgd(model, images, labels, steps, step_size, epsilon):
    #print("images")
    model.eval()
    images_adv = images.detach()
    images_adv = images_adv + 0.001 * torch.randn_like(images_adv)
    images_adv = torch.clamp(images_adv, min=0, max=1)

    for _ in range(steps):
        images_adv.requires_grad_(True)
        loss = F.cross_entropy(model(images_adv), labels)
        grad = torch.autograd.grad(loss, [images_adv])[0].detach()

        images_adv = images_adv.detach() + step_size * torch.sign(grad)
        images_adv = torch.minimum(torch.maximum(images_adv, images - epsilon), images + epsilon)
        images_adv = torch.clamp(images_adv, 0.0, 1.0)

    return images_adv

def FGSM(model, images, labels):
    return pgd(model, images, labels, steps=1, step_size=8/255, epsilon=8/255)
def PGD10(model, images, labels):
    return pgd(model, images, labels, steps=10, step_size=2/255, epsilon=8/255)
def PGD20(model, images, labels):
    return pgd(model, images, labels, steps=20, step_size=1/255, epsilon=8/255)



# @torch.enable_grad()
# def pgd_features(model, features, labels, steps, step_size, epsilon):
#     model.eval()
#     #features_max = torch.max(features.reshape(features.shape[0], -1), 1)[0]
#
#     features_epsilon = features_epsilon.unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1) * torch.ones_like(features)
#
#     features_adv = features.detach()
#     features_adv = features_adv + 0.001 * torch.randn_like(features_adv)
#     #features_adv = torch.clamp(features_adv, min=0, max=1)
#
#     for _ in range(steps):
#         features_adv.requires_grad_(True)
#         loss = F.cross_entropy(model(features_adv), labels)
#         grad = torch.autograd.grad(loss, [features_adv])[0].detach()
#
#         features_adv = features_adv.detach() + step_size * torch.sign(grad)
#         features_adv = torch.minimum(torch.maximum(features_adv, features - features_epsilon), features + features_epsilon)
#         #features_adv = torch.clamp(features_adv, 0.0, 1.0)
#
#     return features_adv

@torch.enable_grad()
def pgd_features(model, features, labels, steps, step_size, epsilon):
    # print("features")
    features_max = torch.max(features.reshape(features.shape[0], -1), 1)[0]
    features_epsilon = epsilon * features_max
    features_epsilon = features_epsilon.unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1) * torch.ones_like(features)
    step_size = step_size / epsilon * features_epsilon
    features_adv = features.detach()
    features_adv = features_adv + 0.001 * torch.randn_like(features_adv)

    for _ in range(steps):
        features_adv.requires_grad_(True)
        loss = F.cross_entropy(model(features_adv), labels)
        grad = torch.autograd.grad(loss, [features_adv])[0].detach()
        features_adv = features_adv.detach() + step_size * torch.sign(grad)
        features_adv = torch.minimum(torch.maximum(features_adv, features - features_epsilon), features + features_epsilon)

    return features_adv


def PGD20_features(model, features, labels, input='image', epsilon=8/255):
    if input == 'feature':
        return pgd_features(model, features, labels, steps=20, step_size=1/255, epsilon=epsilon)
    elif input == 'image':
        return pgd(model, features, labels, steps=20, step_size=1/255, epsilon=epsilon)