import numpy as np
from numpy.linalg import inv
import random
from tqdm import tqdm


# pocket perceptron


class PocketPerceptron:

    def __init__(self, iterations=1000, n_min=50, n_max=200, random_state=42):
        self.iterations = iterations
        self.n_min = n_min
        self.n_max = n_max
        self.random_state = random_state
        self.w = None
        self.best_error_history = []

    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)

        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)

        self.w = np.zeros(X.shape[1])
        best_w = self.w.copy()
        best_error = len(y)
        self.best_error_history = []

        for _ in tqdm(range(self.iterations), desc='Treinando Pocket Perceptron'):
            # sorteia mini-batch aleatório
            n = random.randint(self.n_min, self.n_max)
            idx = np.random.randint(0, len(X), size=n)
            X_batch = X[idx]
            y_batch = y[idx]

            for i in range(n):
                pred = np.sign(np.dot(self.w, X_batch[i]))
                pred = 1 if pred == 0 else pred  # desempate

                if pred != y_batch[i]:
                    # atualiza pesos quando erra
                    self.w = self.w + X_batch[i] * y_batch[i]

                    # guarda o melhor w encontrado até agora
                    novo_erro = self._erro(X_batch, y_batch)
                    if novo_erro < best_error:
                        best_error = novo_erro
                        best_w = self.w.copy()
                        self.best_error_history.append(best_error)

                        if best_error == 0:
                            break

        self.w = best_w
        return self

    def _erro(self, X, y):
        preds = np.sign(X @ self.w)
        preds[preds == 0] = 1
        return np.mean(preds != y)

    def predict(self, X):
        X = np.array(X, dtype=float)
        preds = np.sign(X @ self.w)
        preds[preds == 0] = 1
        return preds

    def get_w(self):
        return self.w

    def set_w(self, w):
        self.w = w



# regressão linear


class LinearRegression:

    def __init__(self):
        self.w = None

    def __str__(self):
        return "Regressão Linear"

    def fit(self, X, y):
        """Solução analítica pela equação normal: w = (X^T X)^{-1} X^T y"""
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        XtX = X.T @ X
        Xty = X.T @ y
        self.w = inv(XtX) @ Xty

    def predict(self, X):
        X = np.array(X, dtype=float)
        return np.sign(X @ self.w)

    def get_w(self):
        return self.w



# regressao logística 


class LogisticRegression:

    def __init__(self, eta=0.01, iterations=1000, batch_size=256):
        """
        eta        : taxa de aprendizado
        iterations : número de iterações do SGD
        batch_size : tamanho do mini-batch
        """
        self.eta = eta
        self.iterations = iterations
        self.batch_size = batch_size
        self.w = None
        self.cost_history = []

    def __str__(self):
        return "Regressão Logística"

    def fit(self, X, y, lamb=0):
        """
        Treina com SGD. lamb é o parâmetro do weight decay (regularização L2),
        que penaliza pesos muito grandes para evitar overfitting.
        """
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        N, d = X.shape

        self.w = np.zeros(d)  # inicialização com zeros, mais estável que aleatório
        self.cost_history = []

        for t in tqdm(range(self.iterations), desc='Treinando Regressão Logística'):
            # mini-batch
            if self.batch_size < N:
                idx = np.random.choice(N, self.batch_size, replace=False)
                X_b = X[idx]
                y_b = y[idx]
            else:
                X_b, y_b = X, y

            N_b = len(X_b)

            z = y_b * (X_b @ self.w)
            z = np.clip(z, -500, 500)  # evita overflow

            sigmoid_z = 1 / (1 + np.exp(-z))

            # gradiente da função de custo logística
            gradiente = -(1 / N_b) * np.sum(
                X_b * y_b.reshape(-1, 1) * (1 - sigmoid_z).reshape(-1, 1),
                axis=0
            )

            if np.linalg.norm(gradiente) < 1e-8:
                print(f"Convergiu na iteração {t}")
                break

            # weight decay + atualização dos pesos
            self.w = self.w * (1 - self.eta * lamb) - self.eta * gradiente

            if t % 100 == 0:
                self.cost_history.append(self._custo(X_b, y_b))

        return self

    def _custo(self, X, y):
        z = y * (X @ self.w)
        z = np.clip(z, -500, 500)
        return -np.mean(np.log(1 / (1 + np.exp(-z))))

    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        z = X @ self.w
        prob = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        return np.where(prob >= 0.5, 1, -1)

    def get_w(self):
        return self.w.copy()

    def set_w(self, w):
        self.w = np.array(w, dtype=np.float64)



# um contra todos


class OneVsAll:

    def __init__(self, modelo, digitos):
        self.modelo = modelo
        self.digitos = digitos
        self.all_w = []

    def execute(self, X, y, **kwargs):
        X_atual = np.array(X, dtype=float)
        y_atual = np.array(y)
        self.all_w = []

        for digito in self.digitos[:-1]:
            y_bin = np.where(y_atual == digito, 1, -1)
            self.modelo.fit(X_atual, y_bin, **kwargs)  # repassa lamb e outros args
            self.all_w.append(self.modelo.get_w())

            mask = y_atual != digito
            X_atual = X_atual[mask]
            y_atual = y_atual[mask]

    def predict_digit(self, X):
        X = np.array(X, dtype=float)
        predicoes = []
        for x in X:
            classificado = False
            for w, digito in zip(self.all_w, self.digitos[:-1]):
                if np.sign(w @ x) == 1:
                    predicoes.append(digito)
                    classificado = True
                    break
            if not classificado:
                predicoes.append(self.digitos[-1])
        return np.array(predicoes)