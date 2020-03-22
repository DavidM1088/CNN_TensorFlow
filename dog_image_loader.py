import requests
from bs4 import BeautifulSoup
import urllib.request
import os
import shutil
import random
import time

class SearchSite:
    def __init__(self, site_name, url, filter, common_term):
        self.url = url
        self.terms = []
        self.filter = filter
        self.site_name = site_name
        self.common_term = common_term

    def set_terms(self, terms):
        self.terms = terms

    def set_category(self, category):
        self.category = category

    def get_search_urls(self):
        tag = '<category>'
        s = self.url.find(tag)
        urls = []
        for term in self.terms:
            s_term = self.category + ' ' + self.common_term + ' '+term
            s_term = s_term.strip()
            url = self.url[0:s] + s_term + self.url[s+len(tag) : len(self.url)]
            urls.append(url)
        return urls

class Scraper:
    categories = set()
    max_images_per_category = 200000
    next_save_image_num = 0

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

    def getCategories(self, category_names):
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
                # if len(self.categories) >= self.max_categories:
                #     break
                breed = str(hr[l+1:])
                if breed.lower() in category_names :
                    self.categories.add(breed)

        self.categories = sorted(self.categories)

    def setup_dir_path(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)

    def isDuplicate(self, category, file_to_check):
        new_file_size = os.path.getsize(file_to_check)
        dirs = [self.directory_train, self.directory_test, self.directory_validate]
        for dir in dirs:
            cat_root = self.directory_root+'/'+dir+'/'+category
            if not os.path.exists(cat_root):
                continue
            dir_contents = os.listdir(cat_root)
            for fle in dir_contents:
                fpath = cat_root+'/'+fle
                if os.path.isfile(fpath):
                    if os.path.getsize(fpath) == new_file_size:
                        return fpath
        return None

    def saveImage(self, site, image_url):
        download_filename = self.directory_root + '/untested.jpg'
        try:
            urllib.request.urlretrieve(image_url, download_filename)
        except:
            print ('cannot load image at url:'+image_url)
            return False

        duplicate = self.isDuplicate(site.category, download_filename)
        if not duplicate is None:
            return False

        r = random.randint(0, 100)
        if r < 20:
            type_dir = self.directory_validate
        else:
            if r < 90:
                type_dir = self.directory_train
            else:
                type_dir = self.directory_test

        keep_file_path = self.directory_root + '/' + type_dir + '/' + site.category
        self.next_save_image_num  += 1
        num = "{:05d}".format(self.next_save_image_num)
        keep_file_name = keep_file_path + '/' + num + '_'+site.site_name+'.jpg'

        if not keep_file_path in self.counts:
            self.counts[keep_file_path] = 0
        self.counts[keep_file_path] += 1
        self.setup_dir_path(keep_file_path)
        os.rename(download_filename, keep_file_name)
        return True

    def getImages_google(self, site,  search_url):
        html_text = requests.get(search_url).text
        soup = BeautifulSoup(html_text, 'html.parser')

        image_ctr = 0
        image_dups = 0
        for img in soup.findAll('img'):
            d = img.attrs
            if not 'src' in img.attrs:
                continue
            img_url = img['src']
            if img_url.find(site.filter) < 0:
                continue
            if self.saveImage(site, img_url):
                image_ctr += 1
            else:
                image_dups += 1
            #time.sleep(0.1)

        print(site.site_name+' images for category:'+site.category+', search str:_[' + search_url + ']_ images:'+str(image_ctr) + " dups:"+str(image_dups))
        #time.sleep(0.1)
        return image_ctr

    def getImages(self, site,  url):
        image_ctr = 0
        image_dups = 0
        html = requests.get(url).text
        split_delim = 'http'
        urls = html.split(split_delim)
        image_ids = []
        for url in urls:
            if url.find(site.filter) < 0:
                continue
            offsets = []
            for delim in [',', '\\', ')', "\""]:
                i = url.find(delim)
                if i >= 0:
                    offsets.append(i)
            if len(offsets) == 0:
                print ("BAD ==="+url)
                continue
            i = min(offsets)
            url = split_delim+url[:i]

            i = url.find('photo-')
            j = url.find('-', i+8)
            id = url[i+6:j]
            if id in image_ids:
                continue
            from urllib.parse import urlparse
            from urllib.parse import parse_qs
            purl = urlparse(url)
            qs = parse_qs(purl.query)
            if not 'ixid' in qs.keys():
                continue
            print (site.site_name, site.category, qs, id, '==\n', url)
            if self.saveImage(site, url):
                image_ctr +=1
            else:
                image_dups += 1
            image_ids.append(id)

        print(site.site_name+' images for category:'+site.category+', search str:_[' + url + ']_ images:'+str(image_ctr) + " dups:"+str(image_dups))

    def getAllImages(self):
        self.counts = {}
        self.image_num = 0
        sites = []

        #unsplash = SearchSite('unsplash', 'https://unsplash.com/s/photos/<category>', 'images.unsplash.com/photo-')
        #unsplash.set_terms([''])
        #sites.append(unsplash)

        google = SearchSite('google', 'https://www.google.com/search?q='+'<category>'+'&source=lnms&tbm=isch', 'http', 'dog picture')
        google.set_terms(['', 'dogtime', 'instagram', 'pinterest', 'facebook', 'breed info', 'typical', 'flickr',
                          'shutterstock', 'american kennel club', 'dog-breeds-expert', 'akc.org', 'dogtime']        )
        sites.append(google)

        for site in sites:
            for category in self.categories:
                site.set_category(category)
                category_image_count = 0
                for url in site.get_search_urls():
                    if site.site_name == 'google':
                        category_image_count += self.getImages_google(site, url)
                    else:
                        self.getImages(site, url)
                print(category+" total images:" + str(category_image_count))


if True:
    g = input("Proceed to delete data?")
    s = Scraper('./images/dogs', 'train','test','validate')
    name_filter = []
    f = open("./config/all_categories.txt", "r")
    for cat in f:
        scat = cat.split(' ')
        if len(scat) > 1:
            print('added:'+scat[0])
            name_filter.append(scat[0])
    s.getCategories(name_filter)
    print (s.categories)
    s.getAllImages()

