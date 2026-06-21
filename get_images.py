import requests
import os
from bs4 import BeautifulSoup
import urllib.parse

CACHE_FOLDER = os.path.join(".", "cache")
DATA_FOLDER = os.path.join(".", "data")

class MyError(Exception):
    pass

def get_png(url: str):
    url = url.removesuffix("/")
    number = os.path.basename(urllib.parse.urlparse(url).path)
    filename = os.path.join(CACHE_FOLDER, f"{number}.html")
    png_filename = os.path.join(DATA_FOLDER, "original", f"{number}.png")

    # check if content is in cache
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            content = f.read()

    # get and save content in cache
    else:
        req = requests.get(url)
        content = req.content
        with open(filename, 'wb') as f:
            f.write(content)

    # parse content
    soup = BeautifulSoup(content, 'html.parser')

    # get div with class pattern_svg
    div = soup.find('object', class_='pattern_svg')
    if not div:
        raise MyError("div not found")

    # get image link
    img = div.find('img')
    if not img:
        raise MyError("img not found")
    png_link = img.get('src')
    if not png_link:
        raise MyError("link not found")

    # don't send request if file already exists
    if os.path.exists(png_filename):
        print("image already saved")
        return
    else:
        # get image from link and save
        req = requests.get(str(png_link))
        with open(png_filename, 'wb') as f:
            f.write(req.content)
        print("saved new image")

if __name__ == "__main__":
    with open("links.txt", "r") as f:
        for line in f:
            try:
                get_png(line.strip())
            except Exception as e:
                print(e)


    