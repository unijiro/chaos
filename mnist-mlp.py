# -*- coding: utf-8 -*-
import time
import numpy as np
import keras
from keras import layers
from keras.datasets import mnist
import matplotlib.pyplot as plt

import SNN

def run_snn(X_train, y_train_oh, X_test, y_test_oh):
    # 電流を毎ステップ一定にしたい場合は encoding="constant"
    snn = SNN.SNN(
        inputnodes=28*28,
        hiddennodes=256,     # 容量アップで精度改善
        outputnodes=10,
        learningrate=0.1,    # readout の学習率
        weight_decay=1e-4,   # 軽いL2正則化
        T=500,               # 時間ステップ（constantなら300〜1000で調整）
        input_scale=50.0,    # 発火率の調整（mean_spike_rate 0.05〜0.3目安）
        encoding="constant", # 一定電流
        # poisson_rate は "poisson" のときのみ利用
        seed=42
    )
    t0 = time.perf_counter()
    snn.train(X_train, y_train_oh, epochs=10, batch_size=256, shuffle=True,
              print_spike_rate=True, print_weight_stats=True)
    acc = snn.evaluate(X_test, y_test_oh)
    elapsed = time.perf_counter() - t0
    print(f"SNN Test Accuracy on MNIST: {acc*100:.2f}% (time {elapsed:.1f}s)")
    return acc, elapsed

def run_mlp(X_train, y_train_oh, X_test, y_test_oh):
    # シンプルな2層MLPベースライン
    model = keras.Sequential([
        keras.Input(shape=(28*28,)),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(256, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    t0 = time.perf_counter()
    model.fit(X_train, y_train_oh, epochs=5, batch_size=256, verbose=2)
    loss, acc = model.evaluate(X_test, y_test_oh, verbose=0)
    elapsed = time.perf_counter() - t0
    print(f"MLP Test Accuracy on MNIST: {acc*100:.2f}% (time {elapsed:.1f}s)")
    return acc, elapsed

def main():
    # MNIST の読み込み
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape((-1, 28*28)).astype('float32') / 255.0
    X_test = X_test.reshape((-1, 28*28)).astype('float32') / 255.0

    # one-hot（両モデルで共通利用）
    y_train_oh = keras.utils.to_categorical(y_train, 10)
    y_test_oh = keras.utils.to_categorical(y_test, 10)

    print("=== Run SNN ===")
    snn_acc, snn_time = run_snn(X_train, y_train_oh, X_test, y_test_oh)

    print("\n=== Run MLP (baseline) ===")
    mlp_acc, mlp_time = run_mlp(X_train, y_train_oh, X_test, y_test_oh)

    print("\n=== Comparison ===")
    print(f"SNN: acc={snn_acc*100:.2f}%  time={snn_time:.1f}s")
    print(f"MLP: acc={mlp_acc*100:.2f}%  time={mlp_time:.1f}s")

if __name__ == "__main__":
    main()