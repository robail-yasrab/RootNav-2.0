import numpy as np
import torch

def _simple_argmax(softmax):
    argmax = torch.argmax(softmax,0)
    mask = np.array(argmax, dtype=np.uint8)
    return mask

class CRF():
    @staticmethod
    def ApplyCRF(model_softmax, image):
        return _simple_argmax(model_softmax)

    @staticmethod
    def decode_channel(mask, index):
        channel_out = np.zeros(mask.shape, dtype=np.uint8)
        if isinstance(index, list):
            for i in index:
                channel_out[mask == i] = 255
        else:
            channel_out[mask == index] = 255

        return channel_out
