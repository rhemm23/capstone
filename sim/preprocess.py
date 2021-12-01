from PIL import Image, UnidentifiedImageError

import numpy as np
import getopt
import sys

if __name__ == '__main__':
  if len(sys.argv) == 1:
    print('Image not specified')
    exit()
  image_path = sys.argv[1]
  sys_args = sys.argv[2:]
  out_path = None
  try:
    args, vals = getopt.getopt(sys_args, 'o:', ['out ='])
    for arg, val in args:
      if arg in ('-o', '--out'):
        out_path = val
  except getopt.error as err:
    print(str(err))
    exit()
  image = None
  try:
    image = Image.open(image_path)
  except UnidentifiedImageError:
    print('Invalid image path')
    exit()

  w, h = image.size

  scale = (300 / h) if (w > h) else (300 / w)

  rh = int(h * scale)
  rw = int(w * scale)

  image = image.resize((rw, rh))
  image = image.convert('L')

  if rh > rw:
    pad = (rh - 300) // 2
    image = image.crop((0, pad, 300, pad + 300))
  elif rw > rh:
    pad = (rw - 300) // 2
    image = image.crop((pad, 0, pad + 300, 300))

  # Save image
  image_bytes = np.asarray(image).tobytes()
  with open(out_path if out_path else 'image.bin', 'wb+') as file:
    file.write(image_bytes)
