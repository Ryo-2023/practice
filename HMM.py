import numpy as np

class Estep:
    def __init__(self,x,A,phi,pi):
        self.x = x
        self.A = A
        self.phi = phi
        self.pi = pi
        self.N = len(self.x)
        self.K = A.shape[0]
    
    def calc_default_alpha(self):
        self.alpha = np.zeros((self.N,self.K))
        for i in range(self.K):
            self.alpha[0,i] = self.pi[i] * self.phi[i,self.x[0]]
    
    def calc_default_beta(self):
        self.beta = np.zeros((self.N,self.K))
        self.beta[self.N-1,:] = 1
        
    def forward_algorithm(self):
        for i in range(1,self.N):
            for j in range(self.K):
                tmp = 0
                for k in range(self.K):
                    tmp += self.alpha[i-1,k] * self.A[k,j]
                self.alpha[i,j] = self.phi[j,self.x[i]] * tmp
        return self.alpha

    def backward_algorithm(self):
        for i in range(self.N - 1,0,-1): # -1のひとつ手前まで→0まで
            for j in range(self.K):
                for k in range(self.K):
                    self.beta[i-1,j] += self.beta[i,j] * self.phi[j,self.x[i]] * self.A[k,j]
        return self.beta

    def calc_Px(self):
        self.forward_algorithm()
        Px = np.sum(self.alpha)
        return Px
    
    def calc_r(self):
        r = np.zeros((self.N,self.K))
        Px = self.calc_Px()
        for i in range(self.N):
            for j in range(self.K):
                r[i,j] = (self.alpha[i,j] * self.beta[i,j]) / Px
        return r
    
    def calc_xi(self):
        Px = self.calc_Px()  #定格化定数
        new_xi = np.zeros((self.N,self.K,self.K))
        for i in range(1,self.N):
            for j in range(self.K):
                for k in range(self.K):
                    new_xi[i,j,k] = self.alpha[i-1, j] * self.phi[j, self.x[i]] * self.A[j, k] * self.beta[i, k]
        return new_xi/Px

class Mstep(Estep):
    def __init__(self,x,A,phi,pi):
        super().__init__(x,A,phi,pi)
        self.calc_default_alpha()
        self.calc_default_beta()
    
    def calc_pi(self):
        new_pi = [0] * self.K
        r = self.calc_r()
        for j in range(self.K):
            sum = 0
            for k in range(self.K):
                sum += r[0,k]
            new_pi[j] = r[0,j] / sum
        return new_pi
    
    def calc_A(self):
        new_A = np.zeros((self.K,self.K))
        xi = self.calc_xi()
        r_sum = np.sum(xi,axis = (0,2))
        for i in range(self.K):
            for j in range(self.K):
                new_A[i,j] = np.sum(xi[:,i,j]) / r_sum[i]
        return new_A
    
    def calc_phi(self):
        new_phi = np.zeros((self.K,self.K))
        r = self.calc_r()
        sum = np.sum(r,axis = 0)
        for k in range(self.K):
            for j in range(self.K):
                for i in range(self.N):
                    new_phi[j,k] += r[i,j] * self.x[k]
        return new_phi/sum
        
def HMM(x,A,phi,pi,epoch):
    Estep_instance = Estep(x,A,phi,pi)
    Mstep_instance = Mstep(x,A,phi,pi)
    final_A = A
    
    final_phi = phi
    final_pi = pi
    Estep_instance.calc_default_alpha()  #初期化
    Estep_instance.calc_default_beta()  #初期化
    
    for _ in range(epoch):
        # Estep
        Estep_instance.forward_algorithm()
        Estep_instance.backward_algorithm()
        Estep_instance.calc_r()
        Estep_instance.calc_xi()
        
        # Mstep
        new_A = Mstep_instance.calc_A()
        new_phi = Mstep_instance.calc_phi()
        new_pi = Mstep_instance.calc_pi()
        
        # update_parameters
        Estep_instance.A = new_A
        Estep_instance.phi = new_phi
        Estep_instance.pi = new_pi
        Mstep_instance.A = new_A
        Mstep_instance.phi = new_phi
        Mstep_instance.pi = new_pi
        
        final_A = new_A
        final_phi = new_phi
        final_pi = new_pi
        #print("new_pi:",new_pi)
        print("sum_gamma",np.sum(Estep_instance.calc_r()))
        
    return final_A, final_phi, final_pi

def calculate_match_rate(true_params, learned_params):
    match_rate = np.mean(np.isclose(true_params, learned_params, atol=1e-2))
    return match_rate

# HMM モデルの学習と評価を行う関数
def test_HMM_learning(x, A, phi, pi, epoch):
    true_A = A
    true_phi = phi
    true_pi = pi
    
    # 学習前のパラメータを表示
    print("True parameters:")
    print("True A:")
    print(true_A)
    print("True phi:")
    print(true_phi)
    print("True pi:")
    print(true_pi)
    
    # HMM モデルの学習を実行
    new_A,new_phi,new_pi = HMM(x, A, phi, pi, epoch)
    
    # 生成されたサンプルデータを表示
    print("\nGenerated sample data:")
    print(x)
    
    # 学習後のパラメータを表示
    print("\nLearned parameters:")
    print("Learned A:")
    print(new_A)
    print("Learned phi:")
    print(new_phi)
    print("Learned pi:")
    print(new_pi)
    
    # パラメータの一致率を計算して表示
    match_rate_A = calculate_match_rate(true_A, new_A)
    match_rate_phi = calculate_match_rate(true_phi, new_phi)
    match_rate_pi = calculate_match_rate(true_pi, new_pi)
    print("\nMatch rate of A:", match_rate_A)
    print("Match rate of phi:", match_rate_phi)
    print("Match rate of pi:", match_rate_pi)

# 真のパラメータを定義
A = np.array([[0.7, 0.3], [0.4, 0.6]])  # 状態遷移確率行列
phi = np.array([[0.8, 0.2], [0.3, 0.7]])  # 観測確率行列
pi = np.array([0.6, 0.4])  # 初期状態確率

# 真のパラメータに基づいてサンプルデータを生成
np.random.seed(0)  # 再現性のために乱数シードを設定
x = [np.random.choice(2, p=pi)]  # 初期状態を選択
for _ in range(1, 5):
    x.append(np.random.choice(2, p=A[x[-1]]))
x = np.array(x)

# HMM モデルの学習と評価を実行
epoch = 100  # エポック数
test_HMM_learning(x, A, phi, pi, epoch)
