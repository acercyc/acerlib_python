from PIL import Image
import os
import glob


def imgConverter(filePatter, convertTo, isKeep=1):
    _, ext = os.path.splitext(filePatter)
    for file in glob.glob(filePatter):
        im = Image.open(file)
        fileNew = file.replace(ext, "." + convertTo)
        im.save(fileNew)
        print(file)
        im.close()

        if not isKeep:
            im.close()
            os.remove(file)

