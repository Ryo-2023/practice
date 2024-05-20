import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PCA_EM:
    def __init__(self,x,m,n):
        # 観測データ
        self.x = x
        # 潜在変数の次元
        self.m = m
        # 出力次元
        self.n = n
        # 入力次元
        self.d = x.shape[0]
        # データ数
        self.N = x.shape[1]
        # 平均
        self.mu = np.zeros((1,self.N))
        for i in range(self.N):
            self.mu[:,i] = np.mean(x[:,i])
        # 共分散行列
        self.S = np.cov(x,rowvar = False,ddof = 0)
        # 固有値,固有ベクトル
        self.eig, self.eig_vec = np.linalg.eigh(self.S)
        # σ^2
        self.sigma = np.sum(self.eig[self.m:self.n-1])/(self.d - self.m)
        # 潜在変数行列
        self.w = np.random.rand(self.d,self.m)
        self.z = np.random.rand(self.m,1)
        # EMアルゴリズムのための初期パラメータ
        self.E_z = np.random.rand(self.N,self.m)
        self.E_zz = np.random.rand(self.N,self.m,self.m)
    
    def e_step(self):
        x = self.x
        x_mean = np.mean(x,axis = 1,keepdims = True)
        w_old = self.w
        sigma_old = self.sigma

        # パラメータ更新に必要な各期待値の計算
        M_old = np.dot(w_old.T ,w_old) + sigma_old * np.eye(self.m)
        E_z = np.dot(np.linalg.inv(M_old) , np.dot(w_old.T , (x - x_mean)))
        E_zz = np.zeros((self.N,self.m,self.m))
        for i in range(self.N):
            E_zz[i,::] = sigma_old * np.linalg.inv(M_old) + np.dot(E_z[:,i],E_z[:,i].T)
        
        # 更新パラメータを格納
        self.E_z = E_z
        self.E_zz = E_zz
    
    def m_step(self):
        x = self.x
        x_mean = np.mean(x,axis = 1)

        # 更新パラメータ計算
        sum_E_z = np.zeros((self.d,self.m))  # sum_E_zの初期化
        for i in range(self.N):
            sum_E_z += np.dot(x[:,i].reshape(-1,1) - x_mean.reshape(-1,1) , self.E_z[:,i].reshape(1,-1))
            
        sum_E_zz = np.zeros_like(self.E_zz[0])  # sum_E_zzの初期化
        for i in range(self.N):
            sum_E_zz += self.E_zz[i,::] 
        new_w = sum_E_z @ np.linalg.pinv(sum_E_zz)

        new_sigma = 0
        for i in range(self.N):
           new_sigma += (np.linalg.norm(x[:,i] - x_mean)**2 - 2 * np.dot(self.E_z[:,i].T , (new_w.T @ (x[:,i] - x_mean)))+ np.trace(np.dot(self.E_zz[i,::], np.dot(new_w.T, new_w))))
        new_sigma = new_sigma / (self.N * self.d)

        # 更新パラメータを格納
        self.w = new_w
        self.sigma = new_sigma
    
    def fit(self,iter):
        for _ in range(iter):
            self.e_step()
            self.m_step()
            
    def gaussian(self,x,mu,sigma):
        x_mu = x
        for i in range(self.N):
            x_mu[:,i] = x_mu[:,i] - mu[:,i]
        return np.exp(-0.5 * np.dot(x_mu.T , np.dot(np.linalg.inv(sigma) , x_mu))) / (np.sqrt(np.linalg.det(sigma)) * (2 * np.pi) ** (self.d / 2))
    
    def transform(self):
        p = self.gaussian(self.x,np.dot(self.w , self.z) + self.mu ,self.sigma * np.eye(self.d))
        return p
    
# データ生成
N = 1000
d = 3
m = 10
n = 2

# 3次元空間に存在するが、その構造が2次元平面上に存在するようなデータを生成
x = np.random.multivariate_normal([0,0,0],[[1,0.5,0.5],[0.5,1,0.5],[0.5,0.5,1]],N).T

# モデルの学習
pca_em = PCA_EM(x, m, n)
pca_em.fit(100)

# データの変換
p = pca_em.transform()

# データの可視化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[0, :], x[1, :], x[2, :])
plt.title('Original data')
plt.grid()
plt.show()

plt.scatter(p[0, :], p[1, :])
plt.title('Transformed data')
plt.grid()
plt.show()
