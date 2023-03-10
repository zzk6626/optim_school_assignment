import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# 设置图片字体
import matplotlib
matplotlib.rc("font", family='Times New Roman')

# ----------感知机模型测试----------- #
# 加载BP网络所需的iris数据集和超参数
x_train = datasets.load_iris().data
y_train = datasets.load_iris().target
n_input = 4
n_hidden = 6
n_output = 3
test_rate = 0.1



# ---------适应度loss函数-------- #
def fit_net(param):
    # 初始化w,b参数
    w1 = param[:n_input * n_hidden].reshape(n_input, n_hidden)
    b1 = param[n_input * n_hidden:n_input * n_hidden + n_hidden].reshape(n_hidden, )
    w2 = param[n_input * n_hidden + n_hidden:n_input * n_hidden + n_hidden + n_hidden * n_output].reshape(n_hidden, n_output)
    b2 = param[n_input * n_hidden + n_hidden + n_hidden * n_output:n_input * n_hidden + n_hidden + n_hidden * n_output + n_output].reshape(n_output)

    # 前向传播
    z1 = x_train.dot(w1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(w2) + b2
    logits = z2

    exp_scores = np.exp(logits)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # 计算loss
    N = x_train.shape[0]
    correct_logprobs = -np.log(probs[range(N), y_train])
    loss = np.sum(correct_logprobs) / N
    return loss

# ------------SSA麻雀搜索算法------------ #
# 对超过边界的变量进行去除
def SSA_Bounds(s, Lb, Ub):
    temp = s
    for i in range(len(s)):
        if temp[i]<Lb[0,i]:
            temp[i]=Lb[0,i]
        elif temp[i]>Ub[0,i]:
            temp[i]=Ub[0,i]
    return temp

# pop是种群，M是迭代次数，fun是用来计算适应度的函数
def SSA(pop, M, c, d, dim):
    P_percent=0.2
    pNum = round(pop*P_percent)  # pNum生产者
    lb = c*np.ones((1,dim))  # lb是下限
    ub = d*np.ones((1,dim))  # ub是上限
    X = np.zeros((pop,dim))  # 麻雀位置
    fit = np.zeros((pop,1))  # 适应度值初始化

    for i in range(pop):
        X[i,:] = lb+(ub-lb)*np.random.rand(1,dim)  # 麻雀属性随机初始化初始值
        fit[i,0] = fit_net(X[i,:])  # 初始化最佳适应度值

    pFit = fit  # 最佳适应度矩阵
    pX = X  # 最佳种群位置
    zp_fit = np.min(fit[:,0])
    bestI = np.argmin(fit[:,0])
    zp_best = X[bestI,:]
    zp_fit_list = []  # 初始化收敛曲线
    for t in range(M): # 迭代更新
        sortIndex = np.argsort(pFit.T)  # 对麻雀的适应度值进行排序，并取出下标
        fmax = np.max(pFit[:,0])
        B = np.argmax(pFit[:,0])
        worse = X[B,:]  # 最差适应度

        r2 = np.random.rand(1) # 预警值
        # 发现者的位置更新
        if r2 < 0.8:
            for i in range(pNum):
                r1 = np.random.rand(1)
                X[sortIndex[0,i],:] = pX[sortIndex[0,i],:]*np.exp(-(i)/(r1*M))
                X[sortIndex[0,i],:] = SSA_Bounds(X[sortIndex[0, i], :], lb, ub)  # 对超过边界的变量进行去除
                fit[sortIndex[0,i],0] = fit_net(X[sortIndex[0, i], :])
        elif r2 >= 0.8:
            for i in range(pNum):
                Q = np.random.rand(1)
                X[sortIndex[0,i],:] = pX[sortIndex[0,i],:]+Q*np.ones((1,dim))
                X[sortIndex[0,i],:] = SSA_Bounds(X[sortIndex[0, i], :], lb, ub)
                fit[sortIndex[0,i],0] = fit_net(X[sortIndex[0, i], :])
        bestII = np.argmin(fit[:,0])
        bestXX = X[bestII,:]

        #  加入者（追随者）的位置更新
        for ii in range(pop-pNum):
            i = ii+pNum
            A = np.floor(np.random.rand(1,dim)*2)*2-1
            if i > pop/2:
                Q = np.random.rand(1)
                X[sortIndex[0,i],:] = Q*np.exp(worse-pX[sortIndex[0,i],:]/np.square(i))
            else:
                X[sortIndex[0,i],:] = bestXX+np.dot(np.abs(pX[sortIndex[0,i],:]-bestXX),1/(A.T*np.dot(A,A.T)))*np.ones((1,dim))
            X[sortIndex[0,i],:] = SSA_Bounds(X[sortIndex[0, i], :], lb, ub)
            fit[sortIndex[0,i],0] = fit_net(X[sortIndex[0, i], :])

        # 意识到危险的麻雀的位置更新
        arrc = np.arange(len(sortIndex[0,:]))
        # 处于种群外围的麻雀向安全区域靠拢，处在种群中心的麻雀随机靠近别的麻雀
        c = np.random.permutation(arrc)
        b = sortIndex[0,c[0:20]]
        for j in range(len(b)):
            if pFit[sortIndex[0,b[j]],0] > zp_fit:
                X[sortIndex[0,b[j]],:] = zp_best+np.random.rand(1,dim)*np.abs(pX[sortIndex[0,b[j]],:]-zp_best)
            else:
                X[sortIndex[0,b[j]],:] = pX[sortIndex[0,b[j]],:]+(2*np.random.rand(1)-1)*np.abs(pX[sortIndex[0,b[j]],:]-worse)/(pFit[sortIndex[0,b[j]]]-fmax+10**(-50))
            X[sortIndex[0,b[j]],:] = SSA_Bounds(X[sortIndex[0, b[j]], :], lb, ub)
            fit[sortIndex[0,b[j]],0] = fit_net(X[sortIndex[0, b[j]]])
        for i in range(pop):
            if fit[i,0] < pFit[i,0]:
                pFit[i,0] = fit[i,0]
                pX[i,:] = X[i,:]
            if pFit[i,0] < zp_fit:
                zp_fit = pFit[i,0]
                zp_best = pX[i,:]
        zp_fit_list.append(zp_fit)
    print("SSA算法最优值：", zp_fit, "SSA算法位置为：", zp_best)
    return zp_fit,zp_best,zp_fit_list

# 随机梯度下降
def BP_net(x_train,y_train,lr,epoch):
    X_train = torch.tensor(x_train,dtype=torch.float)
    Y_train = torch.tensor(y_train).long()
    model = nn.Sequential(
        nn.Linear(4,6),
        nn.Tanh(),
        nn.Linear(6,3)
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=lr)
    loss_list = []
    for i in range(epoch):
        Y_pred = model(X_train)
        loss = loss_fn(Y_pred,Y_train)
        loss_list.append(loss.detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('BP网络误差反向传播算法最优值：',min(loss_list))
    return loss_list

# ------SSA麻雀搜素算法更新参数 w1 b1 w2 b2-------
dim = (4 * 6)  + 6 + (6 * 3) + 3
Max_iteration = 100  # 最大迭代数量
SearchAgents_num = 20   # 麻雀数量
lb = -10    # 最低值
ub = 10    # 最高值
[zp_fit, zp_best, ssa_fit_list] = SSA(SearchAgents_num, Max_iteration, lb, ub, dim)
iter_num = np.arange(len(ssa_fit_list))

# -------BP算法训练更新参数w1 b1 w2 b2-------
lr = 1e-2
loss_list = BP_net(x_train,y_train,lr,epoch=Max_iteration)

# ------fit_value画图----------
plt.plot(iter_num, ssa_fit_list,c='red')
plt.plot(iter_num,loss_list,c='black')
plt.xlabel('Iter_num')
plt.ylabel('Fitness value(loss function)')
plt.title('PerceptronModel_Fitness')
plt.legend(labels=["SSA","BP"],loc="upper right",fontsize=10)
plt.show()
