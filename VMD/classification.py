
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
from time import time


class FastClassifier:
    def __init__(self):
        self.model = models.resnet18(pretrained=True)
        self.model.eval()
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        with open(r'C:\Users\dana\Documents\Ido\follow_up_project\datasets\imagenet_classes.txt') as f:
            self.classes = [line.strip() for line in f.readlines()]

    def apply(self, image, raw=False):
        img_t = self.transform(image)
        batch_t = torch.unsqueeze(img_t, 0)
        out = self.model(batch_t)
        if raw == True:
            return out
        _, index = torch.max(out, 1)
        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
        class_idx =  index[0]
        class_name = self.classes[index[0]]
        cur_percentage = percentage[index[0]].item()
        return class_idx, class_name, cur_percentage


if __name__ == '__main__':

    resnet18 = FastClassifier()
    print(resnet18)

    imagepath = r'C:\Users\dana\Documents\Ido\follow_up_project\benchmark\2019_08_21_MOG2\try_set1\frames_orig\efi_slomo_vid_1_0088.jpg'
    imagepath = r'C:\Users\dana\Documents\Ido\follow_up_project\benchmark\2019_08_21_MOG2\try_set1\frames_orig\efi_slomo_vid_1_0026.jpg'
    image = cv2.imread(imagepath)
    # idxs=(284, 621, 324, 665),(516, 1006, 721, 1083)
    image = image[621:665,284:324,:]

    image = cv2.imread(imagepath)
    det= (426, 742, 477, 795)
    image = image[det[1]:det[3], det[0]:det[2], :]
    # (426, 742, 477, 795)

    cv2.imwrite(r"C:\Users\dana\Documents\Ido\follow_up_project\benchmark\2019_08_21_MOG2\try_set1\try\BB.jpg", image)


    # transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] )])
    start = time()
    class_idx, class_name, cur_percentage = resnet18.apply(image)
    end = time()
    print("inference_time: ", end-start)
    print(class_idx, class_name, cur_percentage)

    out = resnet18.apply(image, raw=True)
    _, index = torch.max(out, 1)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    with open(r'C:\Users\dana\Documents\Ido\follow_up_project\datasets\imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]
    print([(classes[idx], percentage[idx].item()) for idx in indices[0][:5]])