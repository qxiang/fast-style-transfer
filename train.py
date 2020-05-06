import ssl
import torch
import torch.nn.functional as F
import hyperparameters as hp
from torchvision.models import vgg16
from torchvision import transforms
from model import NormalNet

ssl._create_default_https_context = ssl._create_unverified_context


features = []

def _content_loss(input, target):
    return F.mse_loss(input, target.detach())

def _gram(input):
    b, c, w, h = input.size()

    if b == 1:
        input = input.repeat(hp.batch_sz, 1, 1, 1)
        b = hp.batch_sz

    features = input.view(b * c, w * h)
    return torch.mm(features, features.t()).div(b * c * w * h)

def _style_loss(input, target):
    return F.mse_loss(_gram(input), _gram(target).detach())

def _total_loss(input_feats, target_content, target_style):
    content_loss = _content_loss(input_feats[2], target_content[2])

    style_loss = 0
    for in_s, tar_s in zip(input_feats, target_style):
        style_loss += _style_loss(in_s, tar_s)
    
    return content_loss, style_loss

def _first_hook(module, input, output):
    features.clear()
    features.append(output)

def _hook(module, input, output):
    features.append(output)
 
def _hooked_cnn(device):
    cnn = vgg16(pretrained=True).features.to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)
 
    layers_hook = ['3', '8', '15', '22']
    for name, module in cnn.named_modules():
        if name == layers_hook[0]:
            module.register_forward_hook(_first_hook)
        elif name in layers_hook:
            module.register_forward_hook(_hook)

    return torch.nn.Sequential(NormalNet(mean, std).to(device), cnn)


def train_model(model, dataloader, style_img, optimizer, num_epochs, device):
    # Import the vgg model.
    cnn = _hooked_cnn(device)

    cnn(style_img)
    target_style = features.copy()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for i, (img, cap) in enumerate(dataloader):
            img = img.to(device)
            optimizer.zero_grad()
            cnn(img)
            target_content = features.copy()
            new_img = model(img)
            cnn(new_img)
            input_feats = features.copy()
            content_loss, style_loss = _total_loss(input_feats, target_content, target_style)
            content_loss *= hp.content_weight
            style_loss *= hp.style_weight
            loss = content_loss + style_loss
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print("Batch {} <=>Content Loss : {:4f}, Style Loss : {:4f}".format(i, content_loss.item(), style_loss.item()))
                save_image(i, img[0].cpu(), new_img[0].cpu())
                torch.save(model.state_dict(), "checkpoints/checkpoint" + str(i) + ".pt")

def save_image(i, img, new_img):
    postprocess = transforms.Compose([
        transforms.ToPILImage()
    ])

    postprocess(img.view(3, 256, 256)).save("results/original"+ str(i) + ".jpg")
    postprocess(new_img.view(3, 256, 256)).save("results/new"+ str(i) + ".jpg")
