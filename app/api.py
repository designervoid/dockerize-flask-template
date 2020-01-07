from app import app
import os
from random_string import RandomString

obj_random_string = RandomString(20)
obj_random_string.generate_random_string()
image_name = obj_random_string.get_image_name()

@app.route("/api/generate", methods=["POST"])
def generate_text():
    text = request.form["text_from_user"]
    style = request.form["style_from_user"]
    print(text)
    print(image_name)
    os.system(f'python generate.py --noinfo --text="{text}" --filename="{image_name}" --bias=1. --style={style}')
    return jsonify({
        "text_from_user": text,
        "style_from_user": style,
        "image_name": f'{image_name}.png'
    })

@app.route("/api/get_last", methods=["GET"])
def get_last():
    filename = f'imgs/{image_name}.png'
    return send_file(filename, mimetype='image/gif')

@app.route("/api/get/<filename>", methods=["GET"])
def get(filename):
    return send_file(f'imgs/{filename}', mimetype='image/gif')
