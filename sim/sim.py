from weights import *

from ipgu import ipgu

import getopt
import struct
import sys

if __name__ == '__main__':

  img_path = None
  rnw_path = None
  dnw_path = None
  sys_args = sys.argv[1:]

  try:
    args, vals = getopt.getopt(sys_args, 'i:r:d:', ['img =', 'rnw =', 'dnw ='])
    for arg, val in args:
      if arg in ('-i', '--img'):
        img_path = val
      elif arg in ('-r', '--rnw'):
        rnw_path = val
      elif arg in ('-d', '--dnw'):
        dnw_path = val
  except getopt.error as err:
    print(str(err))
    exit()
  if not img_path:
    print('Missing image')
    exit()
  if not rnw_path:
    print('Missing rotational neural weights path')
    exit()
  if not dnw_path:
    print('Missing detection neural weights path')
    exit()

  rnw = read_rnn_weights(rnw_path)
  dnw = read_dnn_weights(dnw_path)
  image = None

  with open(img_path, 'rb') as file:
    image = struct.unpack('B' * 90000, file.read())
  
  img_data = []
  for i in range(300):
    row = []
    for j in range(300):
      row.append(image[(i * 300) + j])
    img_data.append(row)
