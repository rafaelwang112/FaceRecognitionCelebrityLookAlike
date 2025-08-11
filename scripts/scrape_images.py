import os
import requests
from bs4 import BeautifulSoup
import cv2
import numpy as np
import face_recognition

def check_face(binary_data): #check for undetectable faces
    convert = np.frombuffer(binary_data, dtype = np.uint8)
    im = cv2.imdecode(convert, cv2.IMREAD_COLOR)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    detection = face_recognition.face_locations(im)
    return bool(detection)

def get_image(url, path):
    req = requests.get(url)
    soup = BeautifulSoup(req.text, 'html.parser')
    images = soup.find_all('img')
    images = images[:-2] #last 2 aren't people

    for im in images:
        name = im.get('alt')
        link = im.get('src') or im.get('data-src')
        if link.startswith('//'):
            link = 'https:' + link
        image = requests.get(link)
        if not check_face(image.content):
            continue
        with open(os.path.join(path, name + '.jpg'), 'wb') as file:
            file.write(image.content)

    print ("All images downloaded.")


url = 'https://www.thefamouspeople.com/rappers.php'
img_path = "Images"
get_image(url, img_path)
