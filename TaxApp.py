from flask import Flask, request, Response
import base64, pickle, io
from base64 import encodebytes

import numpy as np
import pandas as pd
import os, cv2, glob, math
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import segmentation_models_pytorch as smp

import albumentations as albu
import albumentations.augmentations.geometric.transforms as G
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

from PIL import ImageFile, Image,ImageOps

############################################################

# Initialize the Flask application

app = Flask(__name__)


#device: 'cude' for GPU, otherwise 'cpu'
if torch.cuda == True: DEVICE = 'cuda'
else : DEVICE= 'cpu'




# load best saved checkpoint
best_model = torch.load('new_data_model_v1.pth', map_location=DEVICE)
#defining model encoder
ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


# route http posts to this method
@app.route('/api/test', methods=['POST'])


def test():
    r = request

    # decode image
    image = base64.b64decode(r.json.get('image'))
    image = Image.open(io.BytesIO(image))
    image = np.array(image)
    path = r.json.get('name')
    lat = r.json.get('lat')




    # doing fancy processing here....
    class Dataset(BaseDataset):
        """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

        Args:
            images_dir (str): path to images folder
            masks_dir (str): path to segmentation masks folder
            class_values (list): values of classes to extract from segmentation mask
            augmentation (albumentations.Compose): data transfromation pipeline
                (e.g. flip, scale, etc.)
            preprocessing (albumentations.Compose): data preprocessing
                (e.g. noralization, shape manipulation, etc.)

        """

        def __init__(
                self,
                images,
                augmentation=None,
                preprocessing=None,
        ):

            self.image1 = images
            self.augmentation = augmentation
            self.preprocessing = preprocessing

        def __getitem__(self, i):

            # read data
            image = self.image1
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # apply augmentations
            if self.augmentation:
                sample = self.augmentation(image=image)
                image= sample['image']

            # apply preprocessing
            if self.preprocessing:
                sample = self.preprocessing(image=image)
                image= sample['image']

            return image

        def __len__(self):
            return int(1)



    def get_validation_augmentation():
        """Add paddings to make image shape divisible by 32"""
        test_transform = [
            albu.augmentations.geometric.resize.Resize(504,504),
            G.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=8, pad_width_divisor=8)
        ]
        return albu.Compose(test_transform)


    def to_tensor(x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')


    def get_preprocessing(preprocessing_fn):
        """Construct preprocessing transform

        Args:
            preprocessing_fn (callbale): data normalization function
                (can be specific for each pretrained neural network)
        Return:
            transform: albumentations.Compose

        """

        _transform = [
            albu.Lambda(image=preprocessing_fn),
            albu.Lambda(image=to_tensor),
        ]
        return albu.Compose(_transform)

    # create test dataset
    test_dataset = Dataset(
        image,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )


    test_dataset_vis = Dataset(image,)

    test_dataloader = DataLoader(test_dataset)


    conts = []
    sats = []
    final = []
    final_conts = []
    final_sats = []


    floor = {0: 'Base',
             85: 'Residential',
            170: 'Residential',
            255: 'Commercial'}

    for i in range(len(test_dataset)):
        image_vis = test_dataset_vis[i]
        image = test_dataset[i]

        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        pr_mask2 = np.rollaxis(pr_mask,0,3)
        a,b,c,d = cv2.split(pr_mask2)
        a = a*0
        b = b*85
        c = c*170
        d = d*255
        pred = a+b+c+d
        pred  = cv2.resize(pred, (image_vis.shape[1],image_vis.shape[0]))
        pred = np.resize(pred, (pred.shape[0], pred.shape[1],1))

#         visualize(
#             image=image_vis,
#             predicted_mask=pred
#         )

        pred = cv2.cvtColor(pred,cv2.COLOR_GRAY2RGB)
        sat_image = image_vis
        image = pred.astype("uint8")
        original = image.copy()
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        dilate = cv2.dilate(thresh, kernel, iterations=1)

        # Find contours, obtain bounding box coordinates, and extract ROI
        cnts, heirarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #print(len(cnts))


        image_number = 0
        display = original.copy()
        for number, c in enumerate(cnts):
            x,y,w,h = cv2.boundingRect(c)
            hull = cv2.convexHull(c)
            image = original.copy()
            cv2.drawContours(display, [hull], -1, (0,255,0), 2)
            cv2.drawContours(image, [hull], -1, (0,255,0), -1)
            g2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            r, t2 = cv2.threshold(g2, 100, 200, cv2.THRESH_BINARY)
            masked = cv2.bitwise_and(original, original, mask = t2)
            ROI = masked[y:y+h, x:x+w]

            conts.append(ROI)

            new = Image.fromarray(ROI)
            byte_arr = io.BytesIO()
            new.save(byte_arr, format='PNG') # convert the PIL image to byte array
            encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64

            final_conts.append(encoded_img)

            masked = cv2.bitwise_and(sat_image, sat_image, mask = t2)
            ROI = masked[y:y+h, x:x+w]
            sats.append(ROI)

            new = Image.fromarray(ROI)
            byte_arr = io.BytesIO()
            new.save(byte_arr, format='PNG') # convert the PIL image to byte array
            encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64

            final_sats.append(encoded_img)

        detected = display

    for i, cont in enumerate(conts):
#         image  = cv2.imread(imagename, 0)
        image = cont
#         print("Shape before: ", image.shape)
        hei = max(np.unique(image))

        image_2 = np.where(image!=0, 1, image)


        val = abs((40075016.686 * math.cos(lat))/((2**19)*3500))

#         print("Shape after: ", image_2.shape)

        area  = np.floor(np.sum(image_2)*(val)*10.7639)

        if hei == 85 or hei == 170:
            tax = np.round(area*1.5)
        elif hei == 170:
            tax = np.round(area*5)

        # print("Area covered (sqft): ", area)
        # print("Estimated Tax: PKR", tax)
        BLACK = (255,255,255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 0.3
        font_color = BLACK
        font_thickness = 1
        textH = 'Type: ' + floor[hei]
        textA = 'Area(sqft): ' + str(area)
        textT = 'Tax(PKR): ' + str(tax)

        img_text = sats[i]
        img_text = cv2.copyMakeBorder(img_text,40,40,40,40,cv2.BORDER_CONSTANT,value=[0,0,0,0])
        img_text = cv2.putText(img_text, textA, (1,10), font, font_size, font_color, font_thickness, cv2.LINE_AA)
        img_text = cv2.putText(img_text, textT, (1,25), font, font_size, font_color, font_thickness, cv2.LINE_AA)
        img_text = cv2.putText(img_text, textH, (1,40), font, font_size, font_color, font_thickness, cv2.LINE_AA)


        new = Image.fromarray(img_text)
        byte_arr = io.BytesIO()
        new.save(byte_arr, format='PNG') # convert the PIL image to byte array
        encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64

        final.append(encoded_img)


    new = Image.fromarray(detected)
    byte_arr = io.BytesIO()
    new.save(byte_arr, format='PNG') # convert the PIL image to byte array
    encoded_det = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64


    # done fancy processing....


    return {'images': final, 'conts': final_conts, 'sats': final_sats,'detected': encoded_det}


app.run(host="0.0.0.0", port=5000)