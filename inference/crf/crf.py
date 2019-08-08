import numpy as np
import torch

def _simple_argmax(softmax):
    argmax = torch.argmax(softmax,0)
    mask = np.array(argmax, dtype=np.uint8)
    return mask

class CRF():
    @staticmethod
    def ApplyCRF(model_softmax, image, use_crf):
        mask = None
        if use_crf:
            try:
                import pydensecrf.densecrf as dcrf
                model_softmax = model_softmax.numpy()
                image = np.ascontiguousarray(image)
                unary = -np.log(np.clip(model_softmax,1e-5,1.0))
                c, h, w = unary.shape
                unary = unary.transpose(0, 2, 1)
                unary = unary.reshape(6, -1)
                unary = np.ascontiguousarray(unary)
                d = dcrf.DenseCRF2D(w, h, 6)
                d.setUnaryEnergy(unary)
                d.addPairwiseBilateral(sxy=5, srgb=3, rgbim=image, compat=1)
                q = d.inference(50)
                mask = np.argmax(q, axis=0).reshape(w, h).transpose(1, 0)
            except ImportError:
                # Skip CRF processing if requested but no module available
                mask = _simple_argmax(model_softmax)
        else:
            mask = _simple_argmax(model_softmax)
        return mask

    @staticmethod
    def decode_channel(mask, index):
        channel_out = np.zeros(mask.shape, dtype=np.uint8)
        if isinstance(index, list):
            for i in index:
                channel_out[mask == i] = 255
        else:
            channel_out[mask == index] = 255

        return channel_out
