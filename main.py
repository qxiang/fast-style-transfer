import torch
import torch.optim as optim
import hyperparameters as hp
from torchvision import transforms, datasets
from model import TransNet
from PIL import Image
from train import train_model

def main():
    preprocess = transforms.Compose([
        transforms.Resize((hp.img_size, hp.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]) 

    postprocess = transforms.Compose([
        transforms.Normalize([0, 0, 0], [1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1]),
        transforms.ToPILImage()
    ])

    # Path to folders containing the COCO dataset
    # There are about 80K images in this dataset.
    img_path = "data/coco/images"
    ann_path = "data/coco/annotations/captions_train2014.json"
    style_path = "data/starry-sky.jpg"

    # Set up the device. Use CUDA if it is available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    # Load images into dataset and set up a data loader with batch size 4.
    coco_dataset = datasets.CocoCaptions(img_path, ann_path, transform=preprocess)
    coco_loader = torch.utils.data.DataLoader(coco_dataset, batch_size=hp.batch_sz, 
        shuffle=True, num_workers=4)
    style_img = preprocess(Image.open(style_path)).view(1, 3, hp.img_size, hp.img_size).to(device)

    # Set up the model.
    model = TransNet().to(device)
 
    # Set up the optimizer.
    optimizer = optim.Adam(model.parameters(), lr=hp.learning_rate)

    # Train the model with the COCO dataset.
    train_model(model, coco_loader, style_img, optimizer, hp.num_epochs, device)

    torch.save(model.state_dict(), "checkpoints/model_weights.pt")


if __name__ == "__main__":
    main()
