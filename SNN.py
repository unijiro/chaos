# -*- coding: utf-8 -*-
import numpy as np
import time

# 既定の時間刻みと閾値
DELTA_T: float = 1.0   # [ms]
V_TH: float = 30.0     # [mV]

class IzhikevichLayer:
    """
    ベクトル化 Izhikevich 層（B, H）。
    更新 -> しきい値判定 -> リセット（v=c, u=u+d）。
    既定は Regular Spiking: a=0.02, b=0.2, c=-65, d=8
    """
    def __init__(self, size, a=0.02, b=0.2, c=-65.0, d=8.0, dt=DELTA_T):
        self.size = int(size)
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)
        self.dt = float(dt)
        self.v = None  # (B, H)
        self.u = None  # (B, H)

    def reset_state(self, batch_size):
        v0 = np.full((batch_size, self.size), self.c, dtype=np.float32)
        u0 = self.b * v0
        self.v = v0
        self.u = u0

    def step(self, I):
        """
        I: (B, H) 入力電流
        return: spikes (B, H) 0/1
        """
        v = self.v
        u = self.u
        # 更新
        dv = (0.04 * v * v + 5.0 * v + 140.0 - u + I) * self.dt
        du = (self.a * (self.b * v - u)) * self.dt
        v = v + dv
        u = u + du
        # 更新後にしきい値判定・リセット
        spk = v >= V_TH
        if np.any(spk):
            v = v.copy()
            u = u.copy()
            v[spk] = self.c
            u[spk] = u[spk] + self.d
        self.v = v
        self.u = u
        return spk.astype(np.float32)


def softmax(logits):
    z = logits - np.max(logits, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=1, keepdims=True)


def _stats(arr):
    return dict(
        mean=float(np.mean(arr)),
        std=float(np.std(arr)),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        shape=tuple(arr.shape),
    )


class SNN:
    """
    入力 ->（エンコード）-> 固定ランダム投影 W1 -> Izhikevich 隠れ層（T ステップ）
                                             ↓
                                      スパイク総数 -> 線形 readout (W2, b2) -> softmax
    学習は readout のみ（クロスエントロピー + L2 正則化）。
    """
    name = "Spiking Neural Network (Izhikevich hidden + linear readout)"

    def __init__(
        self,
        inputnodes,
        hiddennodes,
        outputnodes,
        learningrate,
        T=200,
        input_scale=1.0,
        seed=42,
        # Izhikevich params (RS)
        a=0.02, b=0.2, c=-65.0, d=8.0, dt=DELTA_T,
        # 入力エンコーディング
        encoding="constant",       # "constant" or "poisson"
        poisson_rate=200.0,        # [Hz], "poisson"時のみ使用
        # 最適化
        weight_decay=0.0,          # L2 正則化係数（W2に適用）
        print_summary_on_init=True
    ):
        rng = np.random.default_rng(seed)
        self.inputnodes = int(inputnodes)
        self.hiddennodes = int(hiddennodes)
        self.outputnodes = int(outputnodes)
        self.learningrate = float(learningrate)
        self.weight_decay = float(weight_decay)
        self.T = int(T)
        self.input_scale = float(input_scale)
        self.seed = seed
        self.encoding = encoding
        self.poisson_rate = float(poisson_rate)

        # 入力→隠れ（固定ランダム）
        self.W1 = rng.normal(0.0, 1.0 / np.sqrt(self.inputnodes),
                             size=(self.inputnodes, self.hiddennodes)).astype(np.float32)
        self.b1 = np.zeros((self.hiddennodes,), dtype=np.float32)

        # 隠れ→出力（学習対象）
        self.W2 = rng.normal(0.0, 1.0 / np.sqrt(self.hiddennodes),
                             size=(self.hiddennodes, self.outputnodes)).astype(np.float32)
        self.b2 = np.zeros((self.outputnodes,), dtype=np.float32)

        # Izhikevich 隠れ層
        self.hidden_layer = IzhikevichLayer(self.hiddennodes, a=a, b=b, c=c, d=d, dt=dt)

        if print_summary_on_init:
            self.print_summary()

    # パラメータ（ハイパーパラメータと重み統計）の表示
    def summary(self):
        return {
            "model": self.name,
            "seed": self.seed,
            "inputnodes": self.inputnodes,
            "hiddennodes": self.hiddennodes,
            "outputnodes": self.outputnodes,
            "learningrate": self.learningrate,
            "weight_decay": self.weight_decay,
            "time_steps_T": self.T,
            "input_scale": self.input_scale,
            "encoding": self.encoding,
            "poisson_rate": self.poisson_rate,
            "V_TH": V_TH,
            "izhikevich": {
                "a": self.hidden_layer.a,
                "b": self.hidden_layer.b,
                "c": self.hidden_layer.c,
                "d": self.hidden_layer.d,
                "dt": self.hidden_layer.dt,
            },
            "weights": {
                "W1": _stats(self.W1),
                "b1": _stats(self.b1),
                "W2": _stats(self.W2),
                "b2": _stats(self.b2),
            },
        }

    def print_summary(self):
        s = self.summary()
        print("[SNN Summary]")
        for k, v in s.items():
            if k in ("weights", "izhikevich"):
                print(f"  {k}:")
                for kk, vv in v.items():
                    print(f"    {kk}: {vv}")
            else:
                print(f"  {k}: {v}")

    def _forward_hidden_counts_constant(self, X):
        """
        一定入力（時間不変）で T ステップ回す
        """
        B = X.shape[0]
        self.hidden_layer.reset_state(B)
        I_h = X @ self.W1 + self.b1  # (B, H)
        I_h = self.input_scale * I_h
        S_h = np.zeros_like(I_h, dtype=np.float32)
        for _ in range(self.T):
            s_h = self.hidden_layer.step(I_h)
            S_h += s_h
        return S_h

    def _forward_hidden_counts_poisson(self, X):
        """
        Poisson レート符号化で T ステップ回す
        X: (B, D) in [0,1]
        p = X * rate * dt / 1000
        """
        B = X.shape[0]
        D = X.shape[1]
        self.hidden_layer.reset_state(B)
        rng = np.random.default_rng(self.seed)  # 再現性用
        p = X * (self.poisson_rate * self.hidden_layer.dt / 1000.0)  # (B, D)
        p = np.clip(p, 0.0, 1.0)
        S_h = np.zeros((B, self.hiddennodes), dtype=np.float32)
        for _ in range(self.T):
            spikes_in = (rng.random((B, D)) < p).astype(np.float32)  # (B, D)
            I_h_t = spikes_in @ self.W1 + self.b1                    # (B, H)
            I_h_t = self.input_scale * I_h_t
            s_h = self.hidden_layer.step(I_h_t)
            S_h += s_h
        return S_h

    def _forward_hidden_counts(self, X):
        X = X.astype(np.float32)
        if self.encoding == "poisson":
            return self._forward_hidden_counts_poisson(X)
        else:
            return self._forward_hidden_counts_constant(X)

    def logits(self, X):
        S_h = self._forward_hidden_counts(X)
        return S_h @ self.W2 + self.b2

    def predict(self, X, batch_size=256):
        N = X.shape[0]
        preds = []
        for i in range(0, N, batch_size):
            xb = X[i:i+batch_size].astype(np.float32)
            lg = self.logits(xb)
            yhat = np.argmax(lg, axis=1)
            preds.append(yhat)
        return np.concatenate(preds, axis=0)

    def train(
        self,
        training_data,
        targets,
        epochs=5,
        batch_size=128,
        shuffle=True,
        print_spike_rate=True,
        print_weight_stats=True,
    ):
        X = training_data.astype(np.float32)
        if targets.ndim == 1:
            C = self.outputnodes
            y = np.eye(C, dtype=np.float32)[targets]
        else:
            y = targets.astype(np.float32)

        N = X.shape[0]
        idx = np.arange(N)

        for ep in range(epochs):
            t0 = time.perf_counter()
            if shuffle:
                np.random.shuffle(idx)
            total_loss = 0.0
            total_correct = 0
            total_count = 0
            rate_accum = 0.0
            rate_batches = 0

            for i0 in range(0, N, batch_size):
                bidx = idx[i0:i0+batch_size]
                xb = X[bidx]
                yb = y[bidx]

                # 前向き
                S_h = self._forward_hidden_counts(xb)         # (B, H)
                logits = S_h @ self.W2 + self.b2              # (B, C)
                probs = softmax(logits)                       # (B, C)

                # 損失（平均交差エントロピー + L2）
                eps = 1e-12
                ce = -np.sum(yb * np.log(probs + eps)) / yb.shape[0]
                l2 = 0.5 * self.weight_decay * np.sum(self.W2 * self.W2)
                loss = ce + l2
                total_loss += float(loss) * yb.shape[0]

                # 精度
                pred = np.argmax(probs, axis=1)
                true = np.argmax(yb, axis=1)
                total_correct += int(np.sum(pred == true))
                total_count += yb.shape[0]

                # 勾配（出力のみ）
                grad_logits = (probs - yb) / yb.shape[0]      # (B, C)
                grad_W2 = S_h.T @ grad_logits                 # (H, C)
                grad_b2 = np.sum(grad_logits, axis=0)         # (C,)
                # L2
                if self.weight_decay > 0:
                    grad_W2 += self.weight_decay * self.W2

                # SGD 更新
                self.W2 -= self.learningrate * grad_W2
                self.b2 -= self.learningrate * grad_b2

                if print_spike_rate:
                    rate = float(np.mean(S_h)) / max(self.T, 1)  # 1ユニットあたり/ステップ
                    rate_accum += rate
                    rate_batches += 1

            avg_loss = total_loss / total_count
            acc = total_correct / total_count
            elapsed = time.perf_counter() - t0

            msg = f"Epoch {ep+1}: loss={avg_loss:.4f} acc={acc*100:.2f}% time={elapsed:.1f}s"
            if print_spike_rate and rate_batches > 0:
                avg_rate = rate_accum / rate_batches
                msg += f" mean_spike_rate={avg_rate:.4f}"
            if print_weight_stats:
                w2s = _stats(self.W2)
                b2s = _stats(self.b2)
                msg += f" | W2(mean={w2s['mean']:.4f}, std={w2s['std']:.4f}) b2(mean={b2s['mean']:.4f}, std={b2s['std']:.4f})"
            print(msg)

    def evaluate(self, test_data, test_targets, batch_size=256):
        X = test_data.astype(np.float32)
        if test_targets.ndim == 1:
            y_true = test_targets
        else:
            y_true = np.argmax(test_targets, axis=1)
        y_pred = self.predict(X, batch_size=batch_size)
        return float(np.mean(y_pred == y_true))

    # 互換用
    def activate(self, inputs):
        inputs = np.array(inputs, dtype=np.float32, copy=False).reshape(1, -1)
        return self.logits(inputs)