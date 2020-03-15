import requests
from bs4 import BeautifulSoup
import urllib.request
import os
import shutil
import random
import time

class Scraper:
    breeds = set()
    max_categories = 30
    max_images = 100
    image_num = 0

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
                if len(self.breeds) >= self.max_categories:
                    break
                breed = str(hr[l+1:])
                #print('breed:'+breed + " "+str(link))
                names = ['basset-hound', 'australian-cattle-dog', 'samoyed']
                if breed.lower() in names :
                    self.breeds.add(breed)

    def setup_dir(self, data_type, category):
        save_path = self.directory_root + "/"+ data_type + "/" + category
        if not os.path.isdir(save_path):
            #shutil.rmtree(save_path)
            os.mkdir(save_path)
        return save_path

    def getImages(self, search_str):
        for breed in self.breeds:
            url = 'https://www.google.com/search?q='+breed+' '+search_str+'&source=lnms&tbm=isch'
            html_text = requests.get(url).text
            soup = BeautifulSoup(html_text, 'html.parser')

            http_images = 0
            ctr = 0

            for img in soup.findAll('img'):
                img_url = str(img)
                if img_url.find("http") < 0:
                    continue
                ctr += 1
                if ctr > self.max_images:
                    break
                r = random.randint(0, 100)
                to_dir = ""
                if r < 20:
                    to_dir = self.directory_validate
                else:
                    if r < 90:
                        to_dir = self.directory_train
                    else:
                        to_dir = self.directory_test
                to_dir = self.setup_dir(to_dir, breed)
                if not to_dir in self.counts:
                    self.counts[to_dir] = 0
                self.counts[to_dir] += 1
                src = [img['src']]
                num = "{:04d}".format(self.image_num)
                self.image_num += 1
                f_name = to_dir+'/'+num+'.jpg'
                urllib.request.urlretrieve(src[0], f_name)
                time.sleep(0.1)
            print('search:'+ search_str+", breed  "+breed+" saved images:"+str(ctr)+" "+to_dir+' ctr:'+str(ctr))
            #time.sleep(0.1)

    def getAllImages(self):
        self.counts = {}
        self.image_num = 1
        self.getImages('')
        self.getImages('breed info')
        self.getImages('typical')
        self.getImages('from side')
        self.getImages('head')
        self.getImages('from behind')
        self.getImages('playing')
        print ('file counts:')
        for f in self.counts.keys():
            print("  ", f, " ", self.counts[f])

if False:
    g = input("Proceed to delete data?")
    s = Scraper('./images', 'train','test','validate')
    s.getCategories()
    print (s.breeds)
    s.getAllImages()

