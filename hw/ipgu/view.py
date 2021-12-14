from PIL import Image

import sys

if __name__ == '__main__':
  if len(sys.argv) != 2 and len(sys.argv)!=3:
    print('Expected an image path arg')
    exit()
  image = None
  with open(sys.argv[-1], 'rb') as file:
    image = Image.frombytes('L', (int(sys.argv[-2]), int(sys.argv[-2])), file.read())#.resize((sys.argv[-2], sys.argv[-2]) )
  image.show()
