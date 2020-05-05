import ssl
import torch
import torch.nn.functional as F
import hyperparameters as hp
from torchvision.models import vgg16
from torchvision import transforms

ssl._create_default_https_context = ssl._create_unverified_context


features = []

def _content_loss(input, target):
    return F.mse_loss(input, target)

def _gram(input):
    b, c, w, h = input.size()

    if b == 1:
        input = input.repeat(hp.batch_sz, 1, 1, 1)
        b = hp.batch_sz

    features = input.view(b * c, w * h)
    return torch.mm(features, features.t()).div(b * c * w * h)

def _style_loss(input, target):
    return F.mse_loss(_gram(input), _gram(target))

def _total_loss(input_feats, target_feats):
    content_loss = _content_loss(input_feats[2], target_feats[2])

    style_loss = 0
    for in_s, tar_s in zip(input_feats, target_feats):
        style_loss += _style_loss(in_s, tar_s)
    
    return hp.content_weight * content_loss + hp.style_weight * style_loss

def _first_hook(module, input, output):
    features.clear()
    features.append(output)

def _hook(module, input, output):
    features.append(output)
 
def _hooked_cnn(device):
    cnn = vgg16(pretrained=True).features.to(device).eval()
    layers_hook = ['3', '8', '15', '22']
    for name, module in cnn.named_modules():
        if name == layers_hook[0]:
            module.register_forward_hook(_first_hook)
        elif name in layers_hook:
            module.register_forward_hook(_hook)

    return cnn


def train_model(model, dataloader, style_img, optimizer, num_epochs, device):
    # Import the vgg model.
    cnn = _hooked_cnn(device)
    norm_trans = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    cnn(style_img)
    target_feats = features.copy()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for i, (img, cap) in enumerate(dataloader):
            img = img.to(device)
            optimizer.zero_grad()
            cnn(img)
            target_feats[2] = features[2]
            new_img = model(img)
            cnn(new_img)
            input_feats = features.copy()
            loss = _total_loss(input_feats, target_feats)
            loss.backward(retain_graph=True)
            if i % 100 == 0:
                print("Loss : {:4f}".format(loss.item()))
                save_image(i, img[0], new_img[0])
            optimizer.step()

def save_image(i, img, new_img):
    postprocess = transforms.Compose([
        transforms.Normalize([0, 0, 0], [1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1]),
        transforms.ToPILImage()
    ])

    postprocess(img.view(3, 256, 256)).save("result/original"+ str(i) + ".jpg")
    postprocess(new_img.view(3, 256, 256)).save("result/new"+ str(i) + ".jpg")