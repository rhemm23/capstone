from ipgu import ipgu
from bcau import bcau
from heu import heu
from iru import iru
from rnn import rnn
from dnn import dnn

def pipeline(image, rnw, dnw):
  sub_images, _ = ipgu(image)
  results = []
  for sub_image in sub_images:
    heu(sub_image)
    rot = rnn(sub_image, rnw)
    iru(sub_image, rot)
    bcau(sub_image)
    heu(sub_image)
    res = dnn(sub_image, dnw)
    results.append(res)
  return results
