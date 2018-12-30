import os

for dirname, subdirs, files in os.walk('flowers/test'):
    for filename in files:
        if filename.endswith('.jpg'):
            flower = os.path.join(dirname, filename)
            os.system(
                f'python predict.py {flower} checkpoints/vgg19_checkpoint.pth --gpu --top_k 5')
