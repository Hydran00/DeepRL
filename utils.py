import os
import json
import random
import tarfile

import cv2
import clip
import gdown
import torch
import torchvision
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def denormalize(img,x0_norm,y0_norm,x1_norm,y1_norm):
    width = img.shape[1]
    height = img.shape[0]
    # print("den",width,height,sep=" | ")
    x0 = int(x0_norm * width)
    y0 = int(y0_norm * height)
    x1 = int(x1_norm * width)
    y1 = int(y1_norm * height)
    return x0,y0,x1,y1

def draw_bbox(img,x0,y0,x1,y1):
    img = cv2.rectangle(img, (x0, y0), (x1, y1), (255,0,0), 2)
    return img

def compute_iou(agent, ground_truth):
    iou =  torchvision.ops.box_iou( agent, ground_truth)[0].item()
    print("iou : ",iou)
    return iou

class RefCOCOg(Dataset):
    FILE_ID = "1wyyksgdLwnRMC9pQ-vjJnNUn47nWhyMD"
    ARCHIVE_NAME = "refcocog.tar.gz"
    NAME = "refcocog"
    ANNOTATIONS = "annotations/refs(umd).p"
    JSON = "annotations/instances.json"
    IMAGES = "images"
    IMAGE_NAME = "COCO_train2014_{}.jpg"

    def __init__(self, data_dir, split, transform=None):
        self.data_dir = data_dir
        self._check_dataset()
        self.split = split
        self._filter_annotation(
            os.path.join(self.data_dir, self.NAME, self.ANNOTATIONS)
        )
        self._load_json()
        self.transform = transform
        self.model, self.preprocess = clip.load("RN50", device=DEVICE)#,jit=False)
        # checkpoint = torch.load("../RN-50-REFCOCOG.pt")
        
        # Use these 3 lines if you use default model setting(not training setting) of the clip. For example, if you set context_length to 100 since your string is very long during training, then assign 100 to checkpoint['model_state_dict']["context_length"] 
        # checkpoint['model_state_dict']["input_resolution"] = self.model.input_resolution #default is 224
        # checkpoint['model_state_dict']["context_length"] = self.model.context_length # default is 77
        # checkpoint['model_state_dict']["vocab_size"] = self.model.vocab_size 

        # self.model.load_state_dict(checkpoint['model_state_dict'])
        self.index_list = [i for i in range(0,len(self.annotation))]
        random.shuffle(self.index_list)

    def _check_dataset(self):
        if not os.path.exists(os.path.join(self.data_dir, self.ARCHIVE_NAME)):
            if not os.path.exists(self.data_dir):
                os.mkdir(self.data_dir)
            print("Downloading dataset...")
            gdown.download(id=self.FILE_ID)
        if not os.path.exists(os.path.join(self.data_dir, self.NAME)):
            print("Extracting dataset...")
            with tarfile.open(
                os.path.join(self.data_dir, self.ARCHIVE_NAME), "r:gz"
            ) as tar:
                tar.extractall(path=self.data_dir)
        else:
            print("Dataset already extracted")

    def _load_json(self):
        with open(os.path.join(self.data_dir, self.NAME, self.JSON)) as f:
            self.json = json.load(f)
        self.json = pd.DataFrame(self.json["annotations"])

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        # get the random index from shuffled list
        random_index = self.index_list[idx]
        # get line by index
        
        raw = self.annotation.iloc[random_index]
        # get image
        image = self._get_image(raw)
        # get sentences
        
        sentences = self._get_sentences(raw)
        # get bbox

        bboxes = self._get_bboxes(raw)

        # return self._get_vector bboxes and image width and height
        return self._get_vector(image, sentences) , bboxes, image.width, image.height, image, sentences

    def _get_image(self, raw):
        # get image_id
        image_id = raw["image_id"]
        # pad image_id to 12 digits
        image_id = str(image_id).zfill(12)
        # convert image to tensor
        image = Image.open(
            os.path.join(
                self.data_dir, self.NAME, self.IMAGES, self.IMAGE_NAME.format(image_id)
            )
        )
        return image

    def _get_sentences(self, raw):
        # get sentences
        sentences = raw["sentences"]
        # get raw sentences
        sentences = [sentence["raw"] for sentence in sentences]
        return sentences

    def _get_bboxes(self, raw):
        # get ref_id
        id = raw["ann_id"]
        bboxes = self.json[self.json["id"] == id]["bbox"].values[0]
        return bboxes

    def _filter_annotation(self, path):
        self.annotation = pd.read_pickle(path)
        self.annotation = pd.DataFrame(self.annotation)
        self.annotation = self.annotation[self.annotation["split"] == self.split]

    def _get_vector(self, image, sentences):
            image = self.preprocess(image).unsqueeze(0).to(DEVICE)
            text = clip.tokenize(sentences).to(DEVICE)
            with torch.no_grad():
                image_features = self.model.encode_image(image)
                text_features = self.model.encode_text(text)
            text_features = torch.mean(text_features,dim=0).to(DEVICE)
            text_features = text_features.unsqueeze(0).to(DEVICE)
            # bbox = torch.tensor(bbox).unsqueeze(0).to(device)
            # print(f"Image shape: {image_features.shape}, Text shape: {text_features.shape}")
            # Combining embeddings with weighted average
            # out = torch.add(0.4 * image_features ,0.6 * text_features)
            
            # # Combine image and text features and normalize
            product = torch.mul(image_features, text_features)
            power = torch.sign(product)* torch.sqrt(torch.abs(product))
            out = torch.div(power, torch.norm(power, dim=1).reshape(-1, 1))
            out =torch.mean(out,dim=0)
            # print(out.shape, power_out.shape)
            # append bbox
            # print(f"Output shape: {power_out.shape}")
            return out.squeeze(0)