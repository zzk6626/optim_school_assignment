import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置图片字体
import matplotlib
matplotlib.rc("font", family='Times New Roman')

# 选择测试函数
def fun(F,X):
    out = 0
    if F == 'F1':
        out = np.sum(X * X)
    elif F == 'F2':
        out = np.sum(np.abs(X)) + np.prod(np.abs(X))
    elif F == 'F3':
        out = np.max(np.abs(X))
    elif F == 'F4':
        out = np.sum(np.square(np.abs(X + 0.5)))
    elif F == 'F5':    # Rastrigin函数
        out = 20 + X[0] ** 2 + X[1] ** 2 - 10 * (np.cos(2 * np.pi * X[0]) + np.cos(2 * np.pi * X[1]))
    elif F == 'F6':    # Schaffer函数
        out = np.sin(3*np.pi*X[0])**2+((X[0]-1)**2)*(1+np.sin(3*np.pi*X[1])**2)+((X[1]-1)**2)*(1+np.sin(3*np.pi*X[1])**2)
    return out


# --------------PSO粒子群算法-------------- #
# SSA对超过边界的变量进行去除
def PSO_Bounds(s, Lb, Ub):
    temp = s
    for i in range(s.shape[0]):
        if temp[i] < Lb:
            temp[i] = Lb
        elif temp[i] > Ub:
            temp[i] = Ub
        return temp

# N-初始化种群数量个数 dim-搜索空间维度 M-迭代的最大次数 c-最大值 d-最小值
def PSO(w, c1, c2, pop, dim, M, c, d, f):
    r1 = np.random.random()
    r2 = np.random.random()
    p = np.zeros((pop, dim))  # 粒子的初始位置
    v = np.zeros((pop, dim))  # 粒子的初始速度
    gp_best = np.zeros((pop, dim))  # 个体最优值初始化
    zp_best = np.zeros((1, dim))  # 种群最优值
    gp_fit = np.zeros(pop)
    zp_fit = 1e5
    zp_fit_list = []
    lb = c  # lb是下限
    ub = d  # ub是上限

    # 初始化种群
    for i in range(pop):  # 初始化粒子群个数
        for j in range(dim):  # 搜索空间维度 循环结束便初始化第一个例子的速度和位置
            p[i][j] = lb+(ub-lb)*np.random.random()
            v[i][j] = lb+(ub-lb)*np.random.random()
        gp_best[i] = p[i]  # 初始化个体的最优位置
        aim = fun(f, p[i,:])  # 计算个体的适应度值
        gp_fit[i] = aim  # 初始化个体的最优适应度值
        if aim < zp_fit:  # 对个体适应度进行比较，计算出最优的种群适应度
            zp_fit = aim
            zp_best = p[i]

    # 更新粒子的位置与速度
    for t in range(M):  # 在迭代次数M内进行循环
        for i in range(pop):  # 对所有种群进行一次循环
            aim = fun(f, p[i, :])  # 计算一次目标函数的适应度
            if aim < gp_fit[i]:  # 比较适应度大小，将小的负值给个体最优
                gp_fit[i] = aim
                gp_best[i] = p[i]
                if gp_fit[i] < zp_fit:  # 如果是个体最优再将和全体最优进行对比
                    zp_best = p[i]
                    zp_fit = gp_fit[i]
        for i in range(pop):  # 更新粒子的速度和位置
            v[i] = w * v[i] + c1 * r1 * (gp_best[i] - p[i]) + c2 * r2 * (zp_best - p[i])
            v[i] = PSO_Bounds(v[i], lb, ub)
            p[i] = p[i] + v[i]
            p[i] = PSO_Bounds(p[i], lb, ub)
        zp_fit_list.append(zp_fit)
    print("PSO算法最优值：", zp_fit, "PSO算法位置为：", zp_best)
    return zp_fit, zp_best, zp_fit_list


# ------------SSA麻雀搜索算法------------ #
# SSA对超过边界的变量进行去除
def SSA_Bounds(s, Lb, Ub):
    temp = s
    for i in range(len(s)):
        if temp[i]<Lb[0,i]:
            temp[i]=Lb[0,i]
        elif temp[i]>Ub[0,i]:
            temp[i]=Ub[0,i]
    return temp

# pop是种群，M是迭代次数，fun是用来计算适应度的函数
def SSA(pop, M, c, d, dim, f):
    P_percent=0.2
    pNum = round(pop*P_percent)  # pNum生产者
    lb = c*np.ones((1,dim))  # lb是下限
    ub = d*np.ones((1,dim))  # ub是上限
    X = np.zeros((pop,dim))  # 麻雀位置
    fit = np.zeros((pop,1))  # 适应度值初始化

    for i in range(pop):
        X[i,:] = lb+(ub-lb)*np.random.rand(1,dim)  # 麻雀属性随机初始化初始值
        fit[i,0] = fun(f,X[i,:])  # 初始化最佳适应度值

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
                fit[sortIndex[0,i],0] = fun(f,X[sortIndex[0, i], :])
        elif r2 >= 0.8:
            for i in range(pNum):
                Q = np.random.rand(1)
                X[sortIndex[0,i],:] = pX[sortIndex[0,i],:]+Q*np.ones((1,dim))
                X[sortIndex[0,i],:] = SSA_Bounds(X[sortIndex[0, i], :], lb, ub)
                fit[sortIndex[0,i],0] = fun(f,X[sortIndex[0, i], :])
        bestII = np.argmin(fit[:,0])
        bestXX = X[bestII,:]

        # 加入者（追随者）的位置更新
        for ii in range(pop-pNum):
            i = ii+pNum
            A = np.floor(np.random.rand(1,dim)*2)*2-1
            if i > pop/2:
                Q = np.random.rand(1)
                X[sortIndex[0,i],:] = Q*np.exp(worse-pX[sortIndex[0,i],:]/np.square(i))
            else:
                X[sortIndex[0,i],:] = bestXX+np.dot(np.abs(pX[sortIndex[0,i],:]-bestXX),1/(A.T*np.dot(A,A.T)))*np.ones((1,dim))
            X[sortIndex[0,i],:] = SSA_Bounds(X[sortIndex[0, i], :], lb, ub)
            fit[sortIndex[0,i],0] = fun(f,X[sortIndex[0, i], :])

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
            fit[sortIndex[0,b[j]],0] = fun(f,X[sortIndex[0, b[j]]])
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

# 选择测试函数,仅F5、F6可定义为dim=2
F_name = 'F1'
dim = 30  # 搜索空间维度

# 为了对比，最大迭代次数,种群数量，搜索边界全部设置相同
# -----------PSO粒子群算法测试--------- #
c1 = 2  # 全局变量
c2 = 2  # 局部变量
w = 1   # 速度分量
N = 15   # 粒子群数量 / 麻雀数量
max_iter = 100    #  最大迭代次数
lb = -5
ub = 5

pso_fit,pso_best,pso_fit_list = PSO(w,c1,c2,N,dim,max_iter,lb,ub,F_name)

# ---------SSA麻雀搜索算法测试-------- #
ssa_fit,ssa_best,ssa_fit_list = SSA(N,max_iter,lb,ub,dim,F_name)

# 若变量空间维度是2维，画出函数图像
def graph(globalbest,X,f):
    Y = X
    X,Y = np.meshgrid(X,Y)
    fig_data = [X,Y]
    Z = fun(f,fig_data)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, Z,alpha=0.15,cmap='winter')
    plt.show()

if dim == 2:
    fig_x = np.linspace(lb,ub,1000)
    graph(ssa_best,fig_x,F_name)

# 画出适应度值图像
fit_len = np.arange(len(ssa_fit_list))
plt.plot(fit_len,ssa_fit_list,c='red')
plt.plot(fit_len,pso_fit_list,c='blue')
plt.legend(labels=["SSA","PSO"],loc="upper right",fontsize=10)
plt.xlabel('Iterations')
plt.ylabel('Fitness value')
plt.title(F_name+'--Fitness')
plt.show()
pass