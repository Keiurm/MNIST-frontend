import base64
from io import BytesIO
import os
import PIL.ImageOps

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Flaskの初期化
app = Flask(__name__)
CORS(app)


def network(x, test=False):
    """
    Neural Network Consoleから出力されたネットワーク構造のPythonコード
    """
    # Input:x -> 1,28,28
    # Convolution -> 16,22,22
    h = PF.convolution(x, 16, (7, 7), (0, 0), name="Convolution")
    # ReLU
    h = F.relu(h, True)
    # MaxPooling -> 16,11,11
    h = F.max_pooling(h, (2, 2), (2, 2))
    # Convolution_2 -> 30,9,9
    h = PF.convolution(h, 30, (3, 3), (0, 0), name="Convolution_2")
    # MaxPooling_2 -> 30,4,4
    h = F.max_pooling(h, (2, 2), (2, 2))
    # Tanh_2
    h = F.tanh(h)
    # Affine -> 150
    h = PF.affine(h, (150,), name="Affine")
    # ReLU_2
    h = F.relu(h, True)
    # Affine_2 -> 10
    h = PF.affine(h, (10,), name="Affine_2")
    # Softmax
    h = F.softmax(h)
    # CategoricalCrossEntropy -> 1
    # 下記の評価用のレイヤーは推論には不要なのでコメントアウト
    # h = F.categorical_cross_entropy(h, y)
    return h


def remove_transparency(im, bg_colour=(255, 255, 255)):
    """
    カラー画像から透過処理を除去し, 除去後の画像を返す
    """
    # Only process if image has transparency (http://stackoverflow.com/a/1963146)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        # Need to convert to RGBA if LA format due to a bug in PIL (http://stackoverflow.com/a/1963146)
        alpha = im.convert("RGBA").split()[-1]
        # Create a new background image of our matt color.
        # Must be RGBA because paste requires both images have the same format
        # (http://stackoverflow.com/a/8720632  and  http://stackoverflow.com/a/9459208)
        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg
    else:
        return im


# http://127.0.0.1:10000 アクセス時
@app.route("/")
def hello():
    return render_template("index.html")


@app.route("/estimate", methods=["POST"])
def estimate():
    enc_data = request.form["img"]
    dec_data = base64.b64decode(enc_data.split(",")[1])
    img = Image.open(BytesIO(dec_data))
    img.save("pic.png")

    nn.load_parameters("results.nnp")

    x = nn.Variable((1, 1, 28, 28))
    y = network(x, test=True)
    img = Image.open("pic.png")  # 画像を開く
    img = remove_transparency(img)
    img = img.resize((28, 28))  # ネットワーク入力に併せてリサイズ
    img = img.convert("L")  # グレースケール化
    img = PIL.ImageOps.invert(img)
    x.d = np.array(img) * (1.0 / 255.0)

    y.forward()
    tmp = {}

    tmp["estiamtes"] = "{}".format(y.d.argmax(axis=1))
    for num, prob in enumerate(y.d[0]):
        tmp["n" + str(num)] = "{:.6f}%".format(prob * 100)

    return jsonify(tmp)


@app.route("/estimate_file", methods=["POST"])
def estimate_file():
    img = request.files["img"]
    img.save("pic.png")
    nn.load_parameters("results.nnp")

    # ネットワークの入出力用変数を準備
    x = nn.Variable((1, 1, 28, 28))  # 入力
    y = network(x, test=True)  # 出力とネットワーク

    img = Image.open("pic.png")  # 画像を開く
    img = img.resize((28, 28))  # ネットワーク入力に併せてリサイズ
    img = img.convert("L")  # グレースケール化
    x.d = np.array(img) * (1.0 / 255.0)

    # 入力を使って推論開始
    y.forward()
    tmp = {}

    print(y.d)
    tmp["estiamtes"] = "{}".format(y.d.argmax(axis=1))

    for num, prob in enumerate(y.d[0]):
        tmp["n" + str(num)] = "{:.6f}%".format(prob * 100)

    return jsonify(tmp)


if __name__ == "__main__":
    # Flaskを起動し, 80番ポートでアクセスを待つ
    # host=0.0.0.0はサーバ外からのアクセスを許可する設定
    app.run(host="0.0.0.0", port=80, debug=True)
