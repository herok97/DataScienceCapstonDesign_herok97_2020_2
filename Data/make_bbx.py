from xml.etree.ElementTree import parse
import os
dir = './annotation/'
file_list = os.listdir(dir)
with open("annotation.txt", 'w') as f:
    for file in file_list:
        tree = parse(dir + file)
        root = tree.getroot()

        # 가로 길이
        width = int(root.find('size').findtext('width'))
        height = int(root.find('size').findtext('height'))

        f.write(file[:-4] + ' ')
        for object in root.iter("object"):
            f.write(object.find("bndbox").findtext("xmin") + ' ')
            f.write(object.find("bndbox").findtext("ymin") + ' ')
            f.write(object.find("bndbox").findtext("xmax") + ' ')
            f.write(object.find("bndbox").findtext("ymax") + ' ')
        f.write('\n')