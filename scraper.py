import requests
from bs4 import BeautifulSoup
import urllib.request
import os
import shutil
import random

class Scraper:
    breeds = set()
    max_categories = 8
    max_images = 5

    def __init__(self, directory_root, directory_train, directory_validate, directory_test):
        self.directory_root = directory_root
        self.directory_train = directory_train
        self.directory_validate = directory_validate
        self.directory_test = directory_test
        for i in range(0,3):
            if i == 0: save_path = self.directory_root + "/"+ directory_train
            if i == 1: save_path = self.directory_root + "/" + directory_validate
            if i == 2: save_path = self.directory_root + "/" + directory_test
            if os.path.isdir(save_path):
                shutil.rmtree(save_path)
            os.mkdir(save_path)
        pass

    def getCategories(self):
        # find dog subtypes
        url = 'https://dogtime.com/dog-breeds/profiles'
        html_text = requests.get(url).text
        soup = BeautifulSoup(html_text, 'html.parser')
        for link in soup.find_all('a'):
            if not 'class' in link.attrs:
                continue
            hr = link.get('href')
            l = hr.rfind('/')
            if l < len(hr):
                if len(self.breeds) > self.max_categories:
                    break
                b = hr[l+1:]
                #print('breed:'+b + " C:"+cls[0]+" "+str(link))
                self.breeds.add(b)

    def setup_dir(self, data_type, category):
        save_path = self.directory_root + "/"+ data_type + "/" + category
        if os.path.isdir(save_path):
            shutil.rmtree(save_path)
        os.mkdir(save_path)
        return save_path

    def getImages(self):
        for breed in self.breeds:
            url = 'https://www.google.com/search?q='+breed+'&source=lnms&tbm=isch'
            html_text = requests.get(url).text
            soup = BeautifulSoup(html_text, 'html.parser')
            ctr = 0
            for img in soup.findAll('img'):
                imgs = str(img)
                if imgs.find("http") < 0:
                    continue
                ctr += 1
                if ctr > self.max_images:
                    break
                #print(img)
                #print(imgs)
                a = [img['src']]
                #print(a[0])
                num = "{:04d}".format(ctr)
                r = random.randint(0, 100)
                to_dir = ""
                if r < 80:
                    to_dir = self.directory_train
                else:
                    if r<90:
                        to_dir = self.directory_validate
                    else:
                        to_dir = self.directory_test
                to_dir = self.setup_dir(to_dir, breed)
                urllib.request.urlretrieve(a[0], to_dir+'/'+num+'.jpg')
            print("breed  "+breed+" saved images:"+str(ctr)+" "+to_dir)

s = Scraper('./images', 'train','test','validation')

s.getCategories()
print (s.breeds)
s.getImages()

