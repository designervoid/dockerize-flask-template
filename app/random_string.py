import random
import string


class RandomString():
    def __init__(self, string_length):
        self.__string_length = string_length

    def get_image_name(self):
        return self.__image_name

    def set_image_name(self, image_name):
        self.__image_name = image_name

    def generate_random_string(self):
        """Generate a random string of fixed length """
        letters = string.ascii_lowercase
        image_name = ''.join(random.choice(letters) for i in range(self.__string_length))
        self.set_image_name(image_name)
        return image_name
