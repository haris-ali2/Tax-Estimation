import requests 
import base64
from base64 import encodebytes
from PIL import Image
import io
import matplotlib.pyplot as plt
import cv2, glob, os
import pandas as pd
import numpy as np


addr = 'http://localhost:5000'
test_url = addr + '/api/test'


def get_response_image(image):
    byte_arr = io.BytesIO()
    image.save(byte_arr, format='PNG') # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
    return encoded_img


i_folder = 'images/'
lat = 'coords_final_19.txt'
o_folder = 'output/'

paths = [name for name in glob.glob(i_folder+'/*.png') if os.path.isfile(name)]
length = len(paths)

lat_file = pd.read_csv(lat, delimiter = ",", names=['seriel', 'long', 'lat'])


while(length > 0):
    length = length-1
    print(paths[length])
    image = Image.open(paths[length],mode = 'r')
    name = paths[length].split('/')[-1]
    img = int(name.split('.')[0]) - 1
    lat = abs(lat_file.loc[lat_file['seriel'] == img, 'lat']).values[-1]

    # send http request with image and receive response
    response = requests.post(test_url,  json={'image':get_response_image(image),'lat':lat, 'name': name})



    # extract image from response
    finals = response.json()['images']
    detected = response.json()['detected']
    conts = response.json()['conts']
    sats = response.json()['sats']
    
    #img = pickle.loads(img1)
    print(f'image {name}, contours {len(finals)}')
    
    if not os.path.isdir(o_folder + "final/" + name.split('.')[0]):
        os.makedirs(o_folder + "final/" + name.split('.')[0])
        path_final = o_folder + "final/" + name.split('.')[0] + "/"
    
    if not os.path.isdir(o_folder + "sat/" + name.split('.')[0]):
        os.makedirs(o_folder + "sat/" + name.split('.')[0])
        path_sat = o_folder + "sat/" + name.split('.')[0] + "/"
        
    if not os.path.isdir(o_folder + "cont/" + name.split('.')[0]):
        os.makedirs(o_folder + "cont/" + name.split('.')[0])
        path_cont = o_folder + "cont/" + name.split('.')[0] + "/"
        
    if not os.path.isdir(o_folder + "detected"):
        os.makedirs(o_folder + "detected")
        path_det = o_folder + "detected" + "/"
        
        
    for i, img in enumerate(finals):
        print(f'contour {i}')
        img = base64.b64decode(img)  
        img = Image.open(io.BytesIO(img))
        
        if not os.path.isfile(path_final + name):
            cv2.imwrite(path_final + name.split('.')[0] + "_final_{}.png".format(i), np.array(img))
            
        img = base64.b64decode(conts[i])  
        img = Image.open(io.BytesIO(img))
        
        if not os.path.isfile(path_cont + name.split('.')[0] + f"_cont_{i}.png"):
            cv2.imwrite(path_cont + name.split('.')[0] + "_cont_{}.png".format(i), np.array(img))
            
        img = base64.b64decode(sats[i])  
        img = Image.open(io.BytesIO(img))
        
        if not os.path.isfile(path_sat +  name.split('.')[0] + f"_sat_{i}.png"):
            cv2.imwrite(path_sat + name.split('.')[0] + "_sat_{}.png".format(i), np.array(img))
        
        
    img = base64.b64decode(detected)  
    img = Image.open(io.BytesIO(img))
    if not os.path.isfile(path_det + name):
        cv2.imwrite(path_det + name, np.array(img))
