import base64
from io import BytesIO

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import numpy as np

from PIL import Image
import PIL.ImageOps
from flask_cors import CORS
from flask import Flask, render_template, request, jsonify

# Flaskの初期化
app = Flask(__name__)
CORS(app)


def network(x, y=None, test=False):
    """
    Neural Network Consoleから出力されたネットワーク構造のPythonコード
    """
    # Input:x -> 1,28,28
    # Convolution -> 16,22,22
    h = PF.convolution(x, 16, (7, 7), (0, 0), name='Convolution')
    # ReLU
    h = F.relu(h, True)
    # MaxPooling -> 16,11,11
    h = F.max_pooling(h, (2, 2), (2, 2))
    # Convolution_2 -> 30,9,9
    h = PF.convolution(h, 30, (3, 3), (0, 0), name='Convolution_2')
    # MaxPooling_2 -> 30,4,4
    h = F.max_pooling(h, (2, 2), (2, 2))
    # Tanh_2
    h = F.tanh(h)
    # Affine -> 150
    h = PF.affine(h, (150,), name='Affine')
    # ReLU_2
    h = F.relu(h, True)
    # Affine_2 -> 10
    h = PF.affine(h, (10,), name='Affine_2')
    # Softmax
    h = F.softmax(h)
    # CategoricalCrossEntropy -> 1
    # 下記の評価用のレイヤーは推論には不要なのでコメントアウト
    #h = F.categorical_cross_entropy(h, y)
    return h


@app.route('/')
def hello():
    """
    Webアプリ本体
    """
    return render_template("index.html")


def remove_transparency(im, bg_colour=(255, 255, 255)):
    """
    カラー画像から透過処理を除去し, 除去後の画像を返す
    """
    # Only process if image has transparency (http://stackoverflow.com/a/1963146)
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
        # Need to convert to RGBA if LA format due to a bug in PIL (http://stackoverflow.com/a/1963146)
        alpha = im.convert('RGBA').split()[-1]
        # Create a new background image of our matt color.
        # Must be RGBA because paste requires both images have the same format
        # (http://stackoverflow.com/a/8720632  and  http://stackoverflow.com/a/9459208)
        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg
    else:
        return im

@app.route('/estimate', methods=["POST"])
def estimate():
    """
    POSTされた画像データをMNISTのネットワークに入力し
    0〜9のどの数字であるか推論結果をJSON形式で返す
    """

    """
    画像の受信
    """
    # POSTされたデータを取得(base64でエンコードされた画像)
    enc_data = request.form['img']
    # 画像をデコード
    dec_data = base64.b64decode(enc_data.split(',')[1])
    # デコードしたデータを画像化
    img = Image.open(BytesIO(dec_data))

    """
    MNISTのネットワーク準備
    """
    # MNISTのネットワークの重みを読み込み
    nn.load_parameters("results.nnp")

    # 入力変数を準備
    x = nn.Variable((1, 1, 28, 28))
    y = network(x, test=True)

    # 入力画像をネットワークに入力するために処理
    # 透過PNG等の場合に透過処理を除去
    img = remove_transparency(img)
    # グレースケール化
    img = img.convert("L")
    # (28,28) ピクセルにリサイズ
    img = img.resize((28, 28))
    # MNISTは黒背景・白字なので, 白黒反転
    img = PIL.ImageOps.invert(img)

    # 画素値が0.0~1.0になるように正規化して入力としてセット
    x.d = np.array(img) * (1.0 / 255.0)

    # 推論の実行
    y.forward()

    # 推論結果(y.d)から最も確率の高いものを推論結果として取得
    results = "{}".format(y.d.argmax(axis=1)[0])

    # 返却用のデータを準備
    tmp = {"estimates": "{}".format(results)}
    # 各数字の確率も追加
    for ct, item in enumerate(y.d[0].tolist()):
        tmp["n"+str(ct)] = item
    # JSON形式で返す
    return jsonify(tmp)


if __name__ == "__main__":
    # Flaskを起動し, 80番ポートでアクセスを待つ
    app.run(host="0.0.0.0", port=80, debug=True)
