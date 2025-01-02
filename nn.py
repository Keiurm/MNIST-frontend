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
    # BinaryConnectConvolution -> 64,24,24
    h = PF.binary_connect_convolution(
        x, outmaps=64, pad=(0, 0), name="BinaryConnectConvolution", kernel=(5, 5)
    )
    # MaxPooling -> 64,12,12
    h = F.max_pooling(h, kernel=(2, 2), stride=(2, 2))
    # BatchNormalization
    h = PF.batch_normalization(
        h, decay_rate=0.5, eps=0.01, batch_stat=not test, name="BatchNormalization"
    )
    # BinarySigmoid
    h = F.binary_sigmoid(h)
    # BinaryConnectConvolution_2 -> 64,8,8
    h = PF.binary_connect_convolution(
        h, outmaps=64, pad=(0, 0), name="BinaryConnectConvolution_2", kernel=(5, 5)
    )
    # MaxPooling_2 -> 64,4,4
    h = F.max_pooling(h, kernel=(2, 2), stride=(2, 2))
    # BatchNormalization_2
    h = PF.batch_normalization(
        h, decay_rate=0.5, eps=0.01, batch_stat=not test, name="BatchNormalization_2"
    )
    # BinarySigmoid_2
    h = F.binary_sigmoid(h)
    # BinaryConnectAffine -> 512
    h = PF.binary_connect_affine(h, n_outmaps=512, name="BinaryConnectAffine")
    # BatchNormalization_3
    h = PF.batch_normalization(
        h, decay_rate=0.5, eps=0.01, batch_stat=not test, name="BatchNormalization_3"
    )
    # BinarySigmoid_3
    h = F.binary_sigmoid(h)
    # BinaryConnectAffine_2 -> 10
    h = PF.binary_connect_affine(h, n_outmaps=10, name="BinaryConnectAffine_2")
    # BatchNormalization_4
    h = PF.batch_normalization(
        h, decay_rate=0.5, eps=0.01, batch_stat=not test, name="BatchNormalization_4"
    )
    # Softmax
    h = F.softmax(h)
    # CategoricalCrossEntropy -> 1
    # h = F.categorical_cross_entropy(h, y)
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
