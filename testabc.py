from PIL import Image, ImageDraw

with open("image.png", 'rb') as img_file:
        ## To display image using PIL ###
        image = Image.open(img_file)
        