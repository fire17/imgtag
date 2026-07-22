from PIL import Image, ImageOps
import numpy as np, time, glob, os
files=sorted(glob.glob('/tmp/imgs/*.jpg'))
S=256
def timeit(fn,reps=3):
    fn()
    t=time.perf_counter()
    for _ in range(reps): fn()
    return (time.perf_counter()-t)/reps/len(files)*1000  # ms per image
def plain():
    for f in files:
        im=Image.open(f).convert('RGB').resize((S,S),Image.BILINEAR)
        a=np.asarray(im)
def draft():
    for f in files:
        im=Image.open(f); im.draft('RGB',(S,S)); im=im.convert('RGB').resize((S,S),Image.BILINEAR)
        a=np.asarray(im)
def draft_bicubic():
    for f in files:
        im=Image.open(f); im.draft('RGB',(S,S)); im=im.convert('RGB').resize((S,S),Image.BICUBIC)
        a=np.asarray(im)
def thumbnail_reduce():
    for f in files:
        im=Image.open(f); im.draft('RGB',(S,S)); im.thumbnail((S,S),Image.BILINEAR, reducing_gap=2.0)
        a=np.asarray(im.convert('RGB'))
print(f"pillow plain open+resize   : {timeit(plain):7.2f} ms/img  -> {1000/timeit(plain):.0f} img/s 1-thread")
d=timeit(draft); print(f"pillow DRAFT+resize BILINEAR: {d:7.2f} ms/img  -> {1000/d:.0f} img/s 1-thread")
d2=timeit(draft_bicubic); print(f"pillow DRAFT+resize BICUBIC : {d2:7.2f} ms/img -> {1000/d2:.0f} img/s")
d3=timeit(thumbnail_reduce); print(f"pillow DRAFT+thumbnail      : {d3:7.2f} ms/img -> {1000/d3:.0f} img/s")
