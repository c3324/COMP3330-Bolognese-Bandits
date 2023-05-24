

import os
import glob
from icrawler.builtin import GoogleImageCrawler

max_data_amount = 5000 # number of total data to have per class including existing training data at the end of search

# Establish number of inputs for each class so we can search to have equal amounts of images per class
data_folder = 'dataset/seg_train'
max_count = 0 
class_counts = {}
for dir in os.listdir(data_folder):
    class_folder = data_folder + "/" + dir
    file_counter = len(glob.glob1(class_folder, "*.jpg"))
    if file_counter > max_count:
        max_count = file_counter 
    class_counts[dir] = file_counter
    
data_folder = 'dataset/web_crawled'
for dir in os.listdir(data_folder):
    class_folder = data_folder + "/" + dir
    file_counter = len(glob.glob1(class_folder, "*.jpg"))
    if file_counter > max_count:
        max_count = file_counter 
    class_counts[dir] = class_counts[dir] + file_counter



# print(class_counts)
# classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
# search_filters = dict(
#     #size='medium',
#     #licence='commercial',
#     date=((2012, 1, 1), (2023, 1, 1))
# )
# for selected_class in classes:
#     print("Searching class", selected_class, "...")
#     google_crawler = GoogleImageCrawler(
#         feeder_threads = 2,
#         parser_threads = 2,
#         downloader_threads = 8,
#         storage={'root_dir': 'dataset/web_crawled/' + selected_class}
#     )
#     google_crawler.crawl(
#         keyword=selected_class, 
#         max_num = (max_data_amount - class_counts[selected_class]),
#         filters=search_filters,
#         min_size=(128,128)
#     )

import time

# Code adapted from https://towardsdatascience.com/image-scraping-with-python-a96feda8af2d

#import selenium
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# import requests
# from lib import hashlib
# import io
# import pandas as pd
# from bs4 import BeautifulSoup
# from PIL import Image

# DRIVER_PATH = 'webDriver/chromedriver.exe'
# wd = webdriver.Chrome(executable_path=DRIVER_PATH)

# def fetch_image_urls(query:str, max_links_to_fetch:int, wd:webdriver, sleep_between_interactions:int=1):
#     def scroll_to_end(wd):
#         wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
#         time.sleep(sleep_between_interactions)    
    
#     # build the google query
#     search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

#     # load the page
#     wd.get(search_url.format(q=query))

#     image_urls = set()
#     image_count = 0
#     results_start = 0
#     while image_count < max_links_to_fetch:
#         scroll_to_end(wd)

#         # get all image thumbnail results
#         #thumbnail_results = wd.find_elements_by_css_selector("img.Q4LuWd")
#         thumbnail_results = wd.find_element(By.CSS_SELECTOR, "img.Q4LuWd")
#         print(thumbnail_results)
#         number_results = len(thumbnail_results)
        
#         print(f"Found: {number_results} search results. Extracting links from {results_start}:{number_results}")
        
#         for img in thumbnail_results[results_start:number_results]:
#             # try to click every thumbnail such that we can get the real image behind it
#             try:
#                 img.click()
#                 time.sleep(sleep_between_interactions)
#             except Exception:
#                 continue

#             # extract image urls    
#             actual_images = wd.find_elements_by_css_selector('img.n3VNCb')
#             for actual_image in actual_images:
#                 if actual_image.get_attribute('src') and 'http' in actual_image.get_attribute('src'):
#                     image_urls.add(actual_image.get_attribute('src'))

#             image_count = len(image_urls)

#             if len(image_urls) >= max_links_to_fetch:
#                 print(f"Found: {len(image_urls)} image links, done!")
#                 break
#         else:
#             print("Found:", len(image_urls), "image links, looking for more ...")
#             time.sleep(30)
#             return
#             load_more_button = wd.find_element_by_css_selector(".mye4qd")
#             if load_more_button:
#                 wd.execute_script("document.querySelector('.mye4qd').click();")

#         # move the result startpoint further down
#         results_start = len(thumbnail_results)

#     return image_urls

# def persist_image(folder_path:str,url:str):
#     try:
#         image_content = requests.get(url).content

#     except Exception as e:
#         print(f"ERROR - Could not download {url} - {e}")

#     try:
#         image_file = io.BytesIO(image_content)
#         image = Image.open(image_file).convert('RGB')
#         file_path = os.path.join(folder_path,hashlib.sha1(image_content).hexdigest()[:10] + '.jpg')
#         with open(file_path, 'wb') as f:
#             image.save(f, "JPEG", quality=85)
#         print(f"SUCCESS - saved {url} - as {file_path}")
#     except Exception as e:
#         print(f"ERROR - Could not save {url} - {e}")

# def search_and_download(search_term:str,driver_path:str,target_path='./images',number_images=5):
#     target_folder = os.path.join(target_path,'_'.join(search_term.lower().split(' ')))

#     if not os.path.exists(target_folder):
#         os.makedirs(target_folder)

#     with webdriver.Chrome(executable_path=driver_path) as wd:
#         res = fetch_image_urls(search_term, number_images, wd=wd, sleep_between_interactions=0.5)
        
#     for elem in res:
#         persist_image(target_folder,elem)

# print(class_counts)
# classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
# search_filters = dict(
#     #size='medium',
#     #licence='commercial',
#     date=((2012, 1, 1), (2023, 1, 1))
# )
# for selected_class in classes:
#     print("Searching class", selected_class, "...")
#     search_and_download(
#         search_term=selected_class,
#         driver_path=DRIVER_PATH,
#         target_path=('dataset/seb_crawled/' + selected_class)
#     )



# Import API class from pexels_api package
# from pexels_api import API
# import requests
# # Type your Pexels API
# PEXELS_API_KEY = 'fRcr7n9czPBwDBqdW1Xh0jXWinKX81sNTydOwbkmRr3VSzzPDnCR2Hku'
# # Create API object
# api = API(PEXELS_API_KEY)

# print(class_counts)
# classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
# for selected_class in classes:
#     print("Searching class", selected_class, "...")
    
#  # Search five 'kitten' photos
#     api.search('mountain', page=1, results_per_page=(max_count - class_counts[selected_class]))
#     # Get photo entries
#     photos = api.get_entries()
#     # Loop the  photos
#     for photo in photos:
#         url = photo.original
#         response = requests.get(url)
#         open('dataset/web_crawled/'+ selected_class + '/' + str(photo.id) + '.jpg', 'wb').write(response.content)

from bing_image_downloader import downloader
import bbid

print(class_counts)
classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
# filters= dict(
#     license='public'
# )
# for selected_class in classes:
#     limit = max_count - class_counts[selected_class]
#     print("Searching class", selected_class, ". Attempting to print", limit, 'images.')
    
#     downloader.download(selected_class, limit=limit, output_dir=('dataset/web_crawled/'), timeout=20)
#     #bbid()
    
print(class_counts)