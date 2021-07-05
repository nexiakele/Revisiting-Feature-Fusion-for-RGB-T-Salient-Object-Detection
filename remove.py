import os



images = os.listdir('DATA/GT')
for image in images:
    if image.find('pedestrian') > -1: #or image.find('light') > -1:
       path = os.path.join('DATA/GT', image)
       os.remove(path)