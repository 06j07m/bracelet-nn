import requests
import os
from bs4 import BeautifulSoup
import urllib.parse

from utils import MyError

CACHE_FOLDER = os.path.join(".", "cache")
DATA_FOLDER = os.path.join(".", "data")

def get_png(url: str):
    url = url.removesuffix("/")
    number = os.path.basename(urllib.parse.urlparse(url).path)
    filename = os.path.join(CACHE_FOLDER, f"{number}.html")

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
        raise MyError("svg div not found")

    # get image link
    img = div.find('img')
    if not img:
        raise MyError("img not found")
    png_link = img.get('src')
    if not png_link:
        raise MyError("link not found")

    # get div with dimensions
    div2 = soup.find('div', class_='pattern_dimensions')
    if not div2:
        raise MyError("dimensions div not found")

    # get actual dimensions
    data = div2.find('div', class_='data')
    if not data:
        raise MyError("dimensions not found")
    dimensions = data.text
    rows, cols = dimensions.split("x")

    png_filename = os.path.join(DATA_FOLDER, "original", f"{number}-{rows}-{cols}.png")

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


    