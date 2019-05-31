from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import string, random
import keras

import abc
ABC = abc.ABCMeta('ABC', (object,), {'__slots__': ()})

class BasicGenerator(keras.utils.Sequence, ABC):
    def __init__(self, batch_size=128, batches_per_epoch=256):
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch

    @abc.abstractmethod
    def generate_batch(self, batch_size):
        raise NotImplementedError

    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, index):
        random.seed(index + int(random.random()*(10**10)))
        return self.generate_batch(self.batch_size)

class ImageGenerator(BasicGenerator):
    def __init__(self, blur_factor=4, height=20, width=200, font_size=20, flatten = False, *args, **kwargs):
        """
        Initializes an dynamic image generator.

        Arguments:
          blur_factor (optional): Defines the radius of the gaussian blur applied
                                  to the text
          height (optional)     : Defines the height of the text image (in pixels)
          width (optional)      : Defines the width of the text image (in pixels)
          font_size (optional)  : Defines the size of the text on the image image
        """
        self.width = width
        self.height = height
        self.font_size = font_size
        self.blur_factor = blur_factor
        self.character_count = int(self.width / self.font_size)
        self.flatten = flatten
        super(ImageGenerator, self).__init__(*args, **kwargs)

    def create_image(self):
        """
        Creates an image based on the properties stored in the self object.

        Returns:
          A tuple with:
            - A grayscale Image object (as defined in the pillow library) with blur applied.
            - A string containing the text on the image.
            - A grayscale Image object without blur applied.
        """
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", self.font_size )
        except OSError:
            font = ImageFont.truetype("./DejaVuSans.ttf", self.font_size )

        text = ''.join(random.choice(string.ascii_uppercase) for _ in range(self.character_count))
        img = Image.new('L', (self.width, self.height), color=255)
        draw = ImageDraw.Draw(img)

        w, h = draw.textsize(text, font=font)
        print('w {},h {}'.format(w,h))

        draw.text(((self.width-w) / 2,(self.height-h) / 2),text,font=font)

        img_filtered = img.filter(ImageFilter.GaussianBlur(self.blur_factor))

        return img_filtered, text, img

    def generate_batch(self, batch_size):
        """
        Creates a batch of training samples.

        Arguments:
          batch_size (required): The amount of training samples to generate.

        Returns:
          A tuple with:
            - A numpy array of size (batch_size, height, width, 1) containing the 
              image data. Each value is rescaled from 0 -> 255 to 0 -> 1.
            - A list of size character_count, each containing a numpy array of
              size (batch_size, #possible characters). The last dimension contains vectors
              with a single 1 and 0's otherwise. The position of the one denotes the correct
              character.
        """
        inputs = np.empty((batch_size, self.height, self.width, 1))
        outputs = [np.empty((batch_size, len(string.ascii_uppercase))) for j in range(self.character_count)]

        for i in range(batch_size):
            x, Y, x_good = self.create_image()
            inputs[i] = (1 - np.array(x).reshape(self.height, self.width, 1)) / 255.0
            for j in range(self.character_count):
                Y_j = ord(Y[j]) - ord(min(string.ascii_uppercase))
                outputs[j][i] = keras.utils.to_categorical(Y_j, num_classes=len(string.ascii_uppercase))

        if self.flatten:
            flatten_outputs = np.zeros((batch_size,self.character_count,len(string.ascii_uppercase)))
            for o in range(batch_size):
                # define an empty numpy array of 10x26 for each observation
                o_tmp = np.zeros((self.character_count, len(string.ascii_uppercase)))
                for char in range(self.character_count):
                    o_tmp[char,] = outputs[char][o]
                flatten_outputs[o,] = o_tmp
            flatten_outputs = flatten_outputs.reshape((batch_size, self.character_count * len(string.ascii_uppercase)))
            return inputs, flatten_outputs
        
        else:
            return inputs, outputs
