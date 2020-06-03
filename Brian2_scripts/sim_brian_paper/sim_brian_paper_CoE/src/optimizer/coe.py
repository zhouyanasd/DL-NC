# -*- coding: utf-8 -*-
"""
    The optimization methods used for NAS.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""
from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.optimizer.bayesian \
    import BayesianOptimization, UtilityFunction

import time

import numpy as np
import geatpy as ga


class CoE_surrgate():
    def __init__(self, f, f_p, SubCom, ranges, borders, precisions, keys, acq, kappa=2.576, xi=0.0,
                 opt='cma', **gp_params):
        self.f = f # for BO with dict input
        self.f_p = f_p
        self.SubCom = SubCom
        self.FieldDR = ga.crtfld(ranges, borders, precisions)
        self.keys = keys
        self.surrogate = BayesianOptimization(
            f=f,
            pbounds= dict(zip(self.keys, [tuple(x) for x in self.FieldDR.T])), # 此处需要修改到和borders匹配的形式
            random_state=1,
            acq=acq, opt=opt, kappa = kappa, xi =xi, **gp_params
        )

    def surrogate_init(self, init_points, LHS_path=None):
        if LHS_path == None:
            LHS_points = self.surrogate.LHSample(init_points.astype(int), self.surrogate._space.bounds)  # LHS for BO
            fit_init = [self.aimfunc(**self.surrogate._space.array_to_params(x)) for x in # 还是要和GA用的函数形式匹配一下
                        LHS_points]  # evaluated by the real fitness
            for x, eva in zip(LHS_points, fit_init):
                self.surrogate._space.register(x, eva)  # add LHS points to solution space
        else:
            LHS_points, fit_init = self.surrogate.load_LHS(LHS_path)
            for x, eva in zip(LHS_points, fit_init):
                self.surrogate._space.register(x, eva)  # add loaded LHS points to solution space
        self.surrogate._gp.fit(self.optimizer._space.params, self.optimizer._space.target)  # initialize the BO model

    def aimfunc(self, Phen, LegV): # for GA with the LegV input and oupput
        Phen_ = dict(zip(self.keys, Phen))
        return [self.aimfunc(Phen_), LegV]

    def punfunc(self,LegV, FitnV):
        return self.f_p(LegV, FitnV)

    def coe_surrogate_real_templet(self, recopt=0.9, pm=0.1,  MAXGEN=100, NIND=10,
                                   problem='R', maxormin=1, SUBPOP=1, GGAP=0.5,
                                   selectStyle='sus', recombinStyle='xovdp',distribute=True, drawing=0):

        """==========================初始化配置==========================="""
        # GGAP = 0.5  # 因为父子合并后选择，因此要将代沟设为0.5以维持种群规模
        NVAR = self.FieldDR.shape[1]  # 得到控制变量的个数
        # 定义进化记录器，初始值为nan
        pop_trace = (np.zeros((MAXGEN, 1)) * np.nan)
        # 定义变量记录器，记录控制变量值，初始值为nan
        var_trace = (np.zeros((MAXGEN, NVAR)) * np.nan)
        repnum = 0  # 初始化重复个体数为0
        ax = None  # 存储上一帧图形
        """=========================开始遗传算法进化======================="""
        if problem == 'R':
            B = ga.crtrp(1, self.FieldDR)  # 定义初始contex vector
        elif problem == 'I':
            B = ga.crtip(1, self.FieldDR)
        [F_B, LegV_B] = self.aimfuc(B, np.ones((1, 1)))  # 求初代contex vector 的 fitness

        # 初始化各个子种群
        P = []
        ObjV = []
        LegV = []
        for SubCom_i in self.SubCom:
            FieldDR_i = self.FieldDR[:, SubCom_i]
            if problem == 'R':
                P_i = ga.crtrp(NIND, FieldDR_i)  # 生成初始种群
            elif problem == 'I':
                P_i = ga.crtip(NIND, FieldDR_i)
            Chrom = B.copy().repeat(NIND, axis=0)  # 替换contex vector中个体基因
            Chrom[:, SubCom_i] = P_i
            LegV_i = np.ones((NIND, 1))
            [ObjV_i, LegV_i] = self.aimfuc(Chrom, LegV_i)  # 求子问题的目标函数值
            for x, eva in zip(Chrom, ObjV_i):
                self.optimizer._space.register(x, eva) # update the solution space
            self.surrogate._gp.fit(self.optimizer._space.params, self.optimizer._space.target) # update the BO model
            P.append(P_i)
            ObjV.append(ObjV_i)
            LegV.append(LegV_i)  # 生成可行性列向量，元素为1表示对应个体是可行解，0表示非可行解
        gen = 0
        badCounter = 0  # 用于记录在“遗忘策略下”被忽略的代数

        # 开始进化！！
        start_time = time.time()  # 开始计时
        while gen < MAXGEN:
            if badCounter >= 10 * MAXGEN:  # 若多花了10倍的迭代次数仍没有可行解出现，则跳出
                break
            for index, (SubCom_i, P_i, ObjV_i, LegV_i) in enumerate(zip(self.SubCom, P, ObjV, LegV)):
                # 进行遗传算子，生成子代
                FieldDR_i = self.FieldDR[:, SubCom_i]
                SelCh = ga.recombin(recombinStyle, P_i, recopt, SUBPOP)  # 重组
                if problem == 'R':
                    SelCh = ga.mutbga(SelCh, FieldDR_i, pm)  # 变异
                    if distribute == True and repnum > P_i.shape[0] * 0.01:  # 当最优个体重复率高达1%时，进行一次高斯变异
                        SelCh = ga.mutgau(SelCh, FieldDR_i, pm)  # 高斯变异
                elif problem == 'I':
                    SelCh = ga.mutint(SelCh, FieldDR_i, pm)

                Chrom = B.copy().repeat(NIND, axis=0)  # 替换contex vector中个体基因
                Chrom[:, SubCom_i] = SelCh

                LegVSel = np.ones((Chrom.shape[0], 1))  # 初始化育种种群的可行性列向量
                Gauss = []
                for x in Chrom:
                    guess = self.optimizer.guess_fixedpoint(self.util, x) # 估计子种群的适应度
                    Gauss.append(guess)
                Chrom_ = np.array(Chrom)[Gauss.argsort()[0:int(1)]] # 找到估计最好的1个基因
                [ObjVSel, LegVSel] = self.aimfuc(Chrom_, LegVSel)  # 求育种种群的目标函数值
                for x, eva in zip(Chrom, ObjV_i):
                    self.optimizer._space.register(x, eva)  # update the solution space
                    #ToDo:需要更新评估的值到估计的基因中
                self.surrogate._gp.fit(self.optimizer._space.params,
                                       self.optimizer._space.target)  # update the BO model
                # 更新context vector 及其fitness （已经考虑排除不可行解）
                for j, (ObjVSel_j, LegVSel_j) in enumerate(zip(ObjVSel, LegVSel)):
                    if maxormin == 1:
                        if ObjVSel_j < F_B and LegVSel_j == 1:
                            F_B = ObjVSel_j
                            B[0] = Chrom[j, :]
                    if maxormin == -1 and LegVSel_j == 1:
                        if ObjVSel_j > F_B:
                            F_B = ObjVSel_j
                            B[0] = Chrom[j, :]

                # 父子合并
                P_i = np.vstack([P_i, SelCh])
                ObjV_i = np.vstack([ObjV_i, ObjVSel])
                LegV_i = np.vstack([LegV_i, LegVSel])
                # 对合并的种群进行适应度评价
                FitnV = ga.ranking(maxormin * ObjV_i, LegV_i, None, SUBPOP)
                if self.punfunc is not None:
                    FitnV = self.punfunc(LegV_i, FitnV)  # 调用罚函数

                bestIdx = np.argmax(FitnV)  # 获取最优个体的下标
                if LegV_i[bestIdx] != 0:
                    feasible = np.where(LegV_i != 0)[0]  # 排除非可行解
                    repnum = len(np.where(ObjV_i[bestIdx] == ObjV_i)[0])  # 计算最优个体重复数
                    badCounter = 0  # badCounter计数器清零
                else:
                    gen -= 1  # 忽略这一代
                    badCounter += 1
                if distribute == True:  # 若要增强种群的分布性（可能会造成收敛慢）
                    idx = np.argsort(ObjV_i[:, 0], 0)
                    dis = np.diff(ObjV_i[idx, 0]) / (
                                np.max(ObjV_i[idx, 0]) - np.min(ObjV_i[idx, 0]) + 1)  # 差分计算距离的修正偏移量
                    dis = np.hstack([dis, dis[-1]])
                    dis = dis + np.min(dis)  # 修正偏移量+最小量=修正绝对量
                    FitnV[idx, 0] *= np.exp(dis)  # 根据相邻距离修改适应度，突出相邻距离大的个体，以增加种群的多样性
                [P_i, ObjV_i, LegV_i] = ga.selecting(selectStyle, P_i, FitnV, GGAP, SUBPOP, ObjV_i,
                                                     LegV_i)  # 选择个体生成新一代种群
                P[index], ObjV[index], LegV[index] = P_i, ObjV_i, LegV_i

            # 记录进化过程
            pop_trace[gen, 0] = F_B  # 记录当代目标函数的最优值
            var_trace[gen, :] = B[0]  # 记录当代最优的控制变量值
            #         repnum = len(np.where(ObjV_i[bestIdx] == ObjV_i)[0]) # 计算最优个体重复数
            gen += 1
        end_time = time.time()  # 结束计时
        times = end_time - start_time

        # 后处理进化记录器
        delIdx = np.where(np.isnan(pop_trace))[0]
        pop_trace = np.delete(pop_trace, delIdx, 0)
        var_trace = np.delete(var_trace, delIdx, 0)
        if pop_trace.shape[0] == 0:
            raise RuntimeError('error: no feasible solution. (有效进化代数为0，没找到可行解。)')
        # 绘图
        if drawing != 0:
            ga.trcplot(pop_trace, [['种群最优个体目标函数值']])
        # 输出结果
        if maxormin == 1:
            best_gen = np.argmin(pop_trace[:, 0])  # 记录最优种群是在哪一代
            best_ObjV = np.min(pop_trace[:, 0])
        elif maxormin == -1:
            best_gen = np.argmax(pop_trace[:, 0])  # 记录最优种群是在哪一代
            best_ObjV = np.max(pop_trace[:, 0])
        print('最优的目标函数值为：%s' % (best_ObjV))
        print('最优的控制变量值为：')
        for i in range(NVAR):
            print(var_trace[best_gen, i])
        print('有效进化代数：%s' % (pop_trace.shape[0]))
        print('最优的一代是第 %s 代' % (best_gen + 1))
        print('时间已过 %s 秒' % (times))
        # 返回进化记录器、变量记录器以及执行时间
        return [pop_trace, var_trace, times]