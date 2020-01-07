from flask import jsonify, request, send_file
from app import app
from app.random_string import RandomString
from app.generate import generate 


# RANDOM STRING
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

obj_random_string = RandomString(20)
obj_random_string.generate_random_string()
image_name = obj_random_string.get_image_name()


# API

@app.route("/api/generate", methods=["POST"])
def generate_text():
    text_from_user = request.form["text_from_user"]
    style_from_user = request.form["style_from_user"]
    print(text_from_user)
    print(image_name)

    generate(text=text_from_user, filename=image_name, style=style_from_user, bias=1., force=False)

    return jsonify({
        "text_from_user": text_from_user,
        "style_from_user": style_from_user,
        "image_name": f'{image_name}.png'
    })

@app.route("/api/get_last", methods=["GET"])
def get_last():
    filename = f'imgs/{image_name}.png'
    return send_file(filename, mimetype='image/gif')

@app.route("/api/get/<filename>", methods=["GET"])
def get(filename):
    return send_file(f'imgs/{filename}', mimetype='image/gif')
