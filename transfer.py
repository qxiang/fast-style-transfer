import torch
import hyperparameters as hp
import cv2 
import numpy as np
from model import TransNet
from PIL import Image
from torchvision import transforms

def main():
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ]) 

    postprocess = transforms.Compose([
        transforms.ToPILImage()
    ])


    weights_path = "checkpoints/weights.pt"
    input_path = "data/city.mp4"
    output_path = "results/star_city.mp4"

    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (1280, 720))
    
    # Set up the model.
    model = TransNet()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    torch.no_grad()

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            new_frame = transfer(model, frame, preprocess, postprocess)
            out.write(new_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()

def transfer(model, frame, preprocess, postprocess):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = model(preprocess(img).view(1, 3, 256, 256))
    new_frame = cv2.cvtColor(cv2.resize(np.array(postprocess(img.view(3, 256, 256))), (1280, 720)), cv2.COLOR_RGB2BGR)
    return new_frame

if __name__ == "__main__":
    main()