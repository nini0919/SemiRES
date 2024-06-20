import numpy as np

def rle2mask(rle_dict):
    height, width = rle_dict["size"]
    mask = np.zeros(height * width, dtype=np.uint8)

    rle_array = np.array(rle_dict["counts"])
    starts = rle_array[0::2]
    lengths = rle_array[1::2]

    current_position = -1
    for start, length in zip(starts, lengths):
     #   current_position += start
        mask[start-1:start-1 + length] = 1
      #  current_position += length

    mask = mask.reshape((height, width), order='F')
    return mask


def mask2rle(img):
    '''
    Convert mask to rle.
    img: numpy array, 
    1 - mask, 
    0 - background
    
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    if len(runs) % 2 != 0:
        runs = np.append(runs, len(pixels))
    
    runs[1::2] -= runs[::2]
    seg=[]
    
    for x in runs:
        
        seg.append(int(x))
    size=[]
    for x in img.shape:
         size.append(int(x))
    result=dict()
    result['counts']=seg
    result['size']=size
    return result



array = np.array([[True, True, False],
                  [True, True, False],])
array = np.array([[1, 1, 0],
                  [1, 1, 0],])
print(array)
rle_result=mask2rle(array)
print(rle_result)
mask_result=rle2mask(rle_result)
print(mask_result)
