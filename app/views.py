from flask import jsonify, request, send_file
from app import app
from app.random_string import RandomString
from app.generate import generate


@app.route("/api/generate", methods=["POST"])
def generate_text():
    text_from_user = request.form["text_from_user"]
    style_from_user = request.form["style_from_user"]
    print(f'text_from_user from api {text_from_user}')

    obj_random_string = RandomString(20)
    obj_random_string.generate_random_string()
    image_name = obj_random_string.get_image_name()
    print(f'image_name from api {image_name}')

    generate(text=text_from_user, filename=image_name,
             w_style=style_from_user, w_bias=1.)

    return jsonify({
        "text_from_user": text_from_user,
        "style_from_user": style_from_user,
        "image_name": f'{image_name}.png'
    })


@app.route("/api/get/<filename>", methods=["GET"])
def get(filename):
    return send_file(f'imgs/{filename}', mimetype='image/gif')
