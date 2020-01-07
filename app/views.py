from flask import jsonify, request, send_file

import random
import string

import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
from matplotlib import animation
import seaborn
from collections import namedtuple

from app import app


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


# GENERATING



def sample(e, mu1, mu2, std1, std2, rho):
    cov = np.array([[std1 * std1, std1 * std2 * rho],
                    [std1 * std2 * rho, std2 * std2]])
    mean = np.array([mu1, mu2])

    x, y = np.random.multivariate_normal(mean, cov)
    end = np.random.binomial(1, e)
    return np.array([x, y, end])


def split_strokes(points):
    points = np.array(points)
    strokes = []
    b = 0
    for e in range(len(points)):
        if points[e, 2] == 1.:
            strokes += [points[b: e + 1, :2].copy()]
            b = e + 1
    return strokes


def cumsum(points):
    sums = np.cumsum(points[:, :2], axis=0)
    return np.concatenate([sums, points[:, 2:]], axis=1)


def sample_text(sess, args_text, translation, style=None, bias=None, force=None):
    fields = ['coordinates', 'sequence', 'bias', 'e', 'pi', 'mu1', 'mu2', 'std1', 'std2',
              'rho', 'window', 'kappa', 'phi', 'finish', 'zero_states']
    vs = namedtuple('Params', fields)(
        *[tf.compat.v1.get_collection(name)[0] for name in fields]
    )

    text = np.array([translation.get(c, 0) for c in args_text])
    coord = np.array([0., 0., 1.])
    coords = [coord]

    # Prime the model with the author style if requested
    prime_len, style_len = 0, 0
    if style is not None:
        # Priming consist of joining to a real pen-position and character sequences the synthetic sequence to generate
        #   and set the synthetic pen-position to a null vector (the positions are sampled from the MDN)
        style_coords, style_text = style
        prime_len = len(style_coords)
        style_len = len(style_text)
        prime_coords = list(style_coords)
        coord = prime_coords[0] # Set the first pen stroke as the first element to process
        text = np.r_[style_text, text] # concatenate on 1 axis the prime text + synthesis character sequence
        sequence_prime = np.eye(len(translation), dtype=np.float32)[style_text]
        sequence_prime = np.expand_dims(np.concatenate([sequence_prime, np.zeros((1, len(translation)))]), axis=0)

    sequence = np.eye(len(translation), dtype=np.float32)[text]
    sequence = np.expand_dims(np.concatenate([sequence, np.zeros((1, len(translation)))]), axis=0)

    phi_data, window_data, kappa_data, stroke_data = [], [], [], []
    sess.run(vs.zero_states)
    sequence_len = len(args_text) + style_len
    for s in range(1, 60 * sequence_len + 1):
        is_priming = s < prime_len

        print('\r[{:5d}] sampling... {}'.format(s, 'priming' if is_priming else 'synthesis'), end='')

        e, pi, mu1, mu2, std1, std2, rho, \
        finish, phi, window, kappa = sess.run([vs.e, vs.pi, vs.mu1, vs.mu2,
                                               vs.std1, vs.std2, vs.rho, vs.finish,
                                               vs.phi, vs.window, vs.kappa],
                                              feed_dict={
                                                  vs.coordinates: coord[None, None, ...],
                                                  vs.sequence: sequence_prime if is_priming else sequence,
                                                  vs.bias: bias
                                              })

        if is_priming:
            # Use the real coordinate if priming
            coord = prime_coords[s]
        else:
            # Synthesis mode
            phi_data += [phi[0, :]]
            window_data += [window[0, :]]
            kappa_data += [kappa[0, :]]
            # ---
            g = np.random.choice(np.arange(pi.shape[1]), p=pi[0])
            coord = sample(e[0, 0], mu1[0, g], mu2[0, g],
                           std1[0, g], std2[0, g], rho[0, g])
            coords += [coord]
            stroke_data += [[mu1[0, g], mu2[0, g], std1[0, g], std2[0, g], rho[0, g], coord[2]]]

            if not force and finish[0, 0] > 0.8:
                print('\nFinished sampling!\n')
                break

    coords = np.array(coords)
    coords[-1, 2] = 1.

    return phi_data, window_data, kappa_data, stroke_data, coords


def generate(model_path='app/pretrained/model-29',
        text=None,
        filename=None,
        style=None,
        bias=None,
        force=None):
    with open('app/data/translation.pkl', 'rb') as file:
        translation = pickle.load(file)
    rev_translation = {v: k for k, v in translation.items()}
    charset = [rev_translation[i] for i in range(len(rev_translation))]
    charset[0] = ''

    config = tf.compat.v1.ConfigProto(
        device_count={'GPU': 0}
    )
    with tf.compat.v1.Session(config=config) as sess:
        saver = tf.compat.v1.train.import_meta_graph(model_path + '.meta')
        saver.restore(sess, model_path)

        while True:
            if text is not None:
                args_text = text
            else:
                args_text = input('What to generate: ')

            style = None
            if style is not None:
                style = None
                with open('app/data/styles.pkl', 'rb') as file:
                    styles = pickle.load(file)

                if style > len(styles[0]):
                    raise ValueError('Requested style is not in style list')

                style = [styles[0][style], styles[1][style]]

            phi_data, window_data, kappa_data, stroke_data, coords = sample_text(sess, args_text, translation, style, bias)
            strokes = np.array(stroke_data)
            epsilon = 1e-8
            strokes[:, :2] = np.cumsum(strokes[:, :2], axis=0)
            minx, maxx = np.min(strokes[:, 0]), np.max(strokes[:, 0])
            miny, maxy = np.min(strokes[:, 1]), np.max(strokes[:, 1])


            fig, ax = plt.subplots(1, 1)
            for stroke in split_strokes(cumsum(np.array(coords))):
                plt.plot(stroke[:, 0], -stroke[:, 1], color='black')
            ax.set_aspect('equal')
            plt.axis('off')
            fig.savefig(f'app/imgs/{filename}.png', transparent=True)   # save the figure to file
            plt.close(fig)


            if text is not None:
                break


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
