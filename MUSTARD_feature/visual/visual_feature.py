import torch
import PIL
import os
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import pickle
import psutil
import os,datetime,time
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

#extract visual feature
def get_visual_vector(filename):
    print(filename)
    img = PIL.Image.open("./picture/" + filename)
    img_ = transform(img).unsqueeze(0).cuda()
    vector = model(img_)
    return vector

model = models.resnet152(pretrained=False)
model.load_state_dict(torch.load("./model/resnet152-b121ed2d.pth"))
model.eval()

transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),])

model = torch.nn.DataParallel(model, device_ids=[0,1,2,3]).cuda()
filelist=os.listdir("./picture")

features={}
for i in range(len(filelist)):
    visual_feature={}
    filename=filelist[i]
    visual_feature['visual']=get_visual_vector(filename)
    visual_name_noext = filename.split('.')[0]
    features[visual_name_noext] = visual_feature
    torch.cuda.empty_cache()

with open('Feature/visual/visual_features.pkl', 'wb') as f:
    pickle.dump(features, f)



