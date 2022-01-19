from pathlib import Path

import numpy as np
import OpenEXR as exr
import Imath

from PIL import Image

## (i) Important note: This function was adapted from https://gist.github.com/jadarve/de3815874d062f72eaf230a7df41771b
## Slightly modified!
def readEXR(path : Path):
    """Read color + depth data from EXR image file.
    
    Parameters
    ----------
    path : Path
        File path.
        
    Returns
    -------
    img : RGB or RGBA image in float32 format. Each color channel
          lies within the interval [0, 1].
          Color conversion from linear RGB to standard RGB is performed
          internally. See https://en.wikipedia.org/wiki/SRGB#The_forward_transformation_(CIE_XYZ_to_sRGB)
          for more information.
          
    Z : Depth buffer in float32 format or None if the EXR file has no Z channel.
    """
    
    exrfile = exr.InputFile(str(path))
    header = exrfile.header()
    
    dw = header['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    
    channelData = dict()
    
    # convert all channels in the image to numpy arrays
    for c in header['channels']:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)
        
        channelData[c] = C
    
    colorChannels = ['R', 'G', 'B', 'A'] if 'A' in header['channels'] else ['R', 'G', 'B']
    if not 'R' in header['channels'] or not 'G' in header['channels'] or not 'B' in header['channels']:
        img = None
    else:
        img = np.concatenate([channelData[c][...,np.newaxis] for c in colorChannels], axis=2)
    
        # linear to standard RGB
        img[..., :3] = np.where(img[..., :3] <= 0.0031308,
                                12.92 * img[..., :3],
                                1.055 * np.power(img[..., :3], 1 / 2.4) - 0.055)
    
        # sanitize image to be in range [0, 1]
        img = np.where(img < 0.0, 0.0, np.where(img > 1.0, 1, img))
    
    Z = None if 'Z' not in header['channels'] else channelData['Z']
    
    return img, Z

def load_rgb(path : Path):
    """Read an image using Pillow

    Parameters
    ----------
    path : Path
        File path.
        
    Returns
    -------
    img : RGB or RGBA image in float32 format.
    """
    image_ = Image.open(str(path))
    img = np.array(image_.getdata(), dtype=np.float32)
    img = img.reshape((256, 256, -1))
    img = img[:, :, 0:3]
    img = img / 255.0
    return img

def load_mono(path : Path, ch=0):
    """Read an image using Pillow, return n-th channel

    Parameters
    ----------
    path : Path
        File path.
    ch   : int
        optional channel id
        
    Returns
    -------
    channel : channel in float32 format
    """
    channel = load_rgb(path)
    channel = channel[..., ch]
    return channel