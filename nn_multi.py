import os

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import numpy as np
from PIL import Image



def network(x, test=False):
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


if __name__ == "__main__":
    # NNCのネットワークパラメータ読み込み
    # (正確にはネットワーク構造も読み込んでいるが今回は使用しない)
    nn.load_parameters("results.nnp")

    # ネットワークの入出力用変数を準備
    x = nn.Variable((1, 1, 28, 28))  # 入力
    y = network(x, test=True)  # 出力とネットワーク

    # mnistフォルダ内の0.png, 1.png, ... 9.pngで順に検証
    img_dir = "mnist"
    # png画像のみリストアップ
    file_list = [i for i in os.listdir(img_dir) if i.endswith(".png")]

    for item in file_list:
        img = Image.open(img_dir + os.sep + item)  # 画像を開く
        img = img.resize((28, 28))  # ネットワーク入力に併せてリサイズ
        img = img.convert("L")  # グレースケール化
        # ndarray化して, 画素値が0.0〜1.0になるように正規化したものを入力
        x.d = np.array(img) * (1.0 / 255.0)

        # 入力を使って推論開始
        y.forward()

        # 結果を表示
        print("# Input file: {}".format(item))
        # y.d にはMNISTの数字カテゴリ0~9の確率が入っているので
        # 一番大きな確率のカテゴリをargmaxで探す
        print("# Estimate:{}".format(y.d.argmax(axis=1)))
        print("# Result detail:")
        # y.d の推定結果を0, ... 9のそれぞれについて表示
        for num, prob in enumerate(y.d[0]):
            print("  {}: {}".format(num, prob))
