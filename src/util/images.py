import OpenEXR as oexr
import numpy as np
from PIL import Image

def oexr_load_rgb(path : Path):
    file_ = oexr.InputFile(path)
    (r, g, b) = file_.channels("RGB")
    return numpy.array([r, g, b])

def oexr_load_mono(path : Path, ch="R"):
    assert ch in ["R", "G", "B"]
    file_ = oexr.InputFile(path)
    data = file_.channels(ch)
    return numpy.array(data)

def load_rgb(path : Path):
    image_ = Image.open(path)
    return np.array(image_.getdata())

def load_mono(path : Path):
    image_ = Image.open(path)
    return load_rgb(path)[0]