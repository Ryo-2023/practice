import numpy as np

# EM アルゴリズムの実装

def gaussian(x, mean, cov):
    """
    与えられたデータ点 x に対するガウシアン分布の確率密度を計算する関数
    """
    d = len(x)
    coeff = 1.0 / (np.power(2*np.pi, d/2) * np.sqrt(np.linalg.det(cov)))
    exponent = -0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(cov)), (x - mean))
    return coeff * np.exp(exponent)

def initialize_parameters(X, n_components):
    """
    パラメータの初期化
    """
    n_samples, _ = X.shape
    # 平均をランダムに初期化
    means = X[np.random.choice(n_samples, n_components, replace=False)]
    # 共分散行列を単位行列で初期化
    covariances = [np.eye(X.shape[1]) for _ in range(n_components)]
    # 重みを均等に初期化
    weights = np.ones(n_components) / n_components
    return means, covariances, weights

def expectation_step(X, means, covariances, weights):
    """
    Expectation ステップ
    """
    n_samples, _ = X.shape
    n_components = len(weights)
    # 各データポイントに対する各クラスタの所属確率を計算
    resp = np.zeros((n_samples, n_components))
    for i in range(n_samples):
        for j in range(n_components):
            resp[i, j] = weights[j] * gaussian(X[i], means[j], covariances[j])
        resp[i] /= np.sum(resp[i])
    return resp

def maximization_step(X, resp):
    """
    Maximization ステップ
    """
    n_samples, _ = X.shape
    n_components = resp.shape[1]
    # 重みの更新
    weights = np.sum(resp, axis=0) / n_samples
    # 平均の更新
    means = np.dot(resp.T, X) / np.sum(resp, axis=0)[:, np.newaxis]
    # 共分散行列の更新
    covariances = []
    for j in range(n_components):
        diff = X - means[j]
        cov = np.dot(resp[:, j] * diff.T, diff) / np.sum(resp[:, j])
        covariances.append(cov)
    return means, covariances, weights

def gmm_fit(X, n_components, max_iter=100, tol=1e-6):
    """
    GMM の学習
    """
    means, covariances, weights = initialize_parameters(X, n_components)
    prev_log_likelihood = None
    for iteration in range(max_iter):
        # Expectation ステップ
        resp = expectation_step(X, means, covariances, weights)
        # Maximization ステップ
        means, covariances, weights = maximization_step(X, resp)
        # 対数尤度の計算
        log_likelihood = np.sum(np.log(np.sum(resp, axis=1)))
        # 収束判定
        if prev_log_likelihood is not None and np.abs(log_likelihood - prev_log_likelihood) < tol:
            break
        prev_log_likelihood = log_likelihood
    return means, covariances, weights

# ダミーデータの生成
np.random.seed(0)
X = np.concatenate([np.random.randn(100, 2), np.random.randn(100, 2) + [3, 3]])

# GMM の学習
n_components = 2
means, covariances, weights = gmm_fit(X, n_components)

# 学習結果の出力
print("Means:", means)
print("Covariances:", covariances)
print("Weights:", weights)
