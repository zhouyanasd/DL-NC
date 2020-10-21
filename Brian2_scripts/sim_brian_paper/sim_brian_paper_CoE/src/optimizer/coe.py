# -*- coding: utf-8 -*-
"""
    The optimization methods used for NAS.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""
from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.core import BaseFunctions
from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.optimizer.surrogate import create_surrogate

import time

import numpy as np
import geatpy as ga


class CoE_surrogate(BaseFunctions):
    def __init__(self, f, f_p, SubCom, ranges, borders, precisions, keys, random_state,
                 surrogate_type = 'rf' , **surrogate_parameters):
        super().__init__()
        self.f = f # for BO with dict input
        self.f_p = f_p
        self.SubCom = SubCom
        self.FieldDR = ga.crtfld(ranges, borders, list(precisions))
        self.keys = keys
        self.surrogate = create_surrogate(surrogate_type = surrogate_type , f = f, random_state= random_state,
                                          pbounds= dict(zip(self.keys, [tuple(x) for x in self.FieldDR.T])),
                                          **surrogate_parameters)

    def aimfunc(self, Phen, LegV): # for GA with the LegV input and output
        res = []
        for phen in Phen:
            res.append(self.surrogate.probe(phen, lazy=False)) # probe replace f and use space checking and register
        return [np.array(res).reshape(-1,1), LegV]

    def punfunc(self, LegV, FitnV):
        if self.f_p == None:
            return FitnV
        else:
            return self.f_p(LegV, FitnV)

    def coe_surrogate_real_templet(self, recopt=0.9, pm=0.1, MAXGEN=100, NIND=10, init_points = 50,
                                   problem='R', maxormin=1, SUBPOP=1, GGAP=0.5, online = True, eva = 1, interval=1,
                                   selectStyle='sus', recombinStyle='xovdp', distribute=True, LHS_path = None, drawing=0):

        """==========================初始化配置==========================="""
        # GGAP = 0.5  # 因为父子合并后选择，因此要将代沟设为0.5以维持种群规模
        NVAR = self.FieldDR.shape[1]  # 得到控制变量的个数
        # 定义进化记录器，初始值为nan
        pop_trace = (np.zeros((MAXGEN, 1)) * np.nan)
        # 定义变量记录器，记录控制变量值，初始值为nan
        var_trace = (np.zeros((MAXGEN, NVAR)) * np.nan)
        repnum = [0] * len(self.SubCom)  # 初始化重复个体数为0
        self.surrogate.initial_model(init_points = init_points, LHS_path = LHS_path, is_LHS = True, lazy = False)
        ax = None  # 存储上一帧图形
        """=========================开始遗传算法进化======================="""
        if problem == 'R':
            B = ga.crtrp(1, self.FieldDR)  # 定义初始context vector
        elif problem == 'I':
            B = ga.crtip(1, self.FieldDR)
        [F_B, LegV_B] = self.aimfunc(B, np.ones((1, 1)))  # 求初代context vector 的 fitness

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
            Chrom = B.copy().repeat(NIND, axis=0)  # 替换context vector中个体基因
            Chrom[:, SubCom_i] = P_i
            LegV_i = np.ones((NIND, 1))
            [ObjV_i, LegV_i] = self.aimfunc(Chrom, LegV_i)  # 求子问题的目标函数值
            self.surrogate.update_model()  # update the surrogate model
            P.append(P_i)
            ObjV.append(ObjV_i)
            LegV.append(LegV_i)  # 生成可行性列向量，元素为1表示对应个体是可行解，0表示非可行解
        gen = 0
        badCounter = 0  # 用于记录在“遗忘策略下”被忽略的代数

        # 开始进化！！
        start_time = time.time()  # 开始计时
        estimation = interval-1  # counter and make sure the first time could be evaluated
        while gen < MAXGEN:
            if badCounter >= 10 * MAXGEN:  # 若多花了10倍的迭代次数仍没有可行解出现，则跳出
                break
            estimation += 1
            for index, (SubCom_i, P_i, ObjV_i, LegV_i) in enumerate(zip(self.SubCom, P, ObjV, LegV)):
                # 进行遗传算子，生成子代
                FieldDR_i = self.FieldDR[:, SubCom_i]
                SelCh = ga.recombin(recombinStyle, P_i, recopt, SUBPOP)  # 重组
                if problem == 'R':
                    SelCh = ga.mutbga(SelCh, FieldDR_i, pm)  # 变异
                    if distribute == True and repnum[index] > P_i.shape[0] * 0.01:  # 当最优个体重复率高达1%时，进行一次高斯变异
                        SelCh = ga.mutgau(SelCh, FieldDR_i, pm)  # 高斯变异
                elif problem == 'I':
                    SelCh = ga.mutint(SelCh, FieldDR_i, pm)

                Chrom = B.copy().repeat(NIND, axis=0)  # 替换contex vector中个体基因
                Chrom[:, SubCom_i] = SelCh
                LegVSel = np.ones((Chrom.shape[0], 1))  # 初始化育种种群的可行性列向量
                ObjVSel = self.surrogate.predict(Chrom).reshape(-1, 1)  # get the estimated value

                guess = self.surrogate.guess(Chrom)  # 估计子种群的(acquisition) function value

                if estimation >= interval: # 设置一个更新间隔
                    best_guess = guess.argsort()[0:int(eva)] # 找到估计最好的eva个基因序号

                    Chrom_ = np.array(Chrom)[best_guess]  # 找到估计最好的eva个基因
                    LegVSel_ = np.ones((Chrom_.shape[0], 1))  # 初始化实际评估种群的可行性列向量
                    [ObjVSel_, LegVSel_] = self.aimfunc(Chrom_, LegVSel_)  # 求育种种群的目标函数值
                    if online :
                        self.surrogate.update_model()  # update the BO model

                    ObjVSel[best_guess] = ObjVSel_  # replace the estimated value by real value
                    LegVSel[best_guess] = LegVSel_

                    # 更新context vector 及其fitness （已经考虑排除不可行解）
                    for j, (ObjVSel_j, LegVSel_j) in enumerate(zip(ObjVSel_, LegVSel_)):
                        if maxormin == 1:
                            if ObjVSel_j < F_B and LegVSel_j == 1:
                                F_B = ObjVSel_j
                                B[0] = Chrom_[j, :]
                        if maxormin == -1 and LegVSel_j == 1:
                            if ObjVSel_j > F_B:
                                F_B = ObjVSel_j
                                B[0] = Chrom_[j, :]

                # 父子合并
                P_i = np.vstack([P_i, SelCh])
                ObjV_i = np.vstack([ObjV_i, ObjVSel])
                LegV_i = np.vstack([LegV_i, LegVSel])
                # 对合并的种群进行适应度评价
                FitnV = ga.ranking(maxormin * ObjV_i, LegV_i, None, SUBPOP)
                FitnV = self.punfunc(LegV_i, FitnV)  # 调用罚函数

                bestIdx = np.argmax(FitnV)  # 获取最优个体的下标
                if LegV_i[bestIdx] != 0:
                    # feasible = np.where(LegV_i != 0)[0]  # 排除非可行解
                    repnum[index] = len(np.where(ObjV_i[bestIdx] == ObjV_i)[0])  # 计算最优个体重复数
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

            if estimation >= interval:
                estimation = 0  # initilize the counter

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


class CoE_surrogate_mixgentype(CoE_surrogate):
    '''
    codes和scales中不需要编码的部分用None来代替。
    '''
    def __init__(self,f, f_p, SubCom, ranges, borders, precisions, codes, scales, keys, random_state,
                 surrogate_type = 'rf', **surrogate_parameters):
        super().__init__(f, f_p, SubCom, ranges, borders, precisions, keys, random_state,
                         surrogate_type = surrogate_type, **surrogate_parameters)
        self.ranges = ranges
        self.borders = borders
        self.precisions = precisions
        self.codes = codes
        self.scales = scales

    def b_coding(self, P_i, SubCom_i):
        NIND = len(P_i)
        if (np.array(self.codes[SubCom_i]) != None).any():
            index_ib = np.where(np.array(self.codes[SubCom_i]) != None)[0]
            codes_ib = self.codes[SubCom_i][index_ib]
            scales_ib = self.scales[SubCom_i][index_ib]
            ranges_ib = self.ranges[:,SubCom_i][:,index_ib]
            borders_ib = self.borders[:,SubCom_i][:,index_ib]
            precisions_ib = self.precisions[SubCom_i][index_ib]
            FieldD = ga.crtfld(ranges_ib, borders_ib, list(precisions_ib), list(codes_ib), list(scales_ib))
            if not self.is2(FieldD):
                raise Exception('worng range of binary coding')
            Lind = np.sum(FieldD[0, :])  # 种群染色体长度
            P_ib = ga.crtbp(NIND, Lind)  # 生成初始种
            variable = ga.bs2rv(P_ib, FieldD)  # 解码
            P_i[:, index_ib] = variable
            return P_i
        else:
            return P_i

    def remu(self, P_i, SubCom_i,recombinStyle,recopt,SUBPOP,pm,distribute,repnum,index):
        FieldDR_i = self.FieldDR[:, SubCom_i]
        SelCh = np.zeros(P_i.shape)
        if (np.array(self.codes[SubCom_i]) != None).any():
            index_ib = np.where(np.array(self.codes[SubCom_i]) != None)[0]
            codes_ib = self.codes[SubCom_i][index_ib]
            scales_ib = self.scales[SubCom_i][index_ib]
            ranges_ib = self.ranges[:,SubCom_i][:,index_ib]
            borders_ib = self.borders[:,SubCom_i][:,index_ib]
            precisions_ib = self.precisions[SubCom_i][index_ib]
            FieldD = ga.crtfld(ranges_ib, borders_ib, list(precisions_ib), list(codes_ib), list(scales_ib))
            if not self.is2(FieldD):
                raise Exception('worng range of binary coding')
            P_ib = self.rv2bs(P_i[:, index_ib],FieldD)
            SelCh_ib = ga.recombin(recombinStyle, P_ib, recopt, SUBPOP)  # 重组
            SelCh_ib = ga.mutbin(SelCh_ib, pm)  # 变异
            variable = ga.bs2rv(SelCh_ib, FieldD)  # 解码

            index_ir = np.where(np.array(self.codes[SubCom_i]) == None)[0]
            P_ir = P_i[:, index_ir]
            FieldDR_ir = FieldDR_i[:, index_ir]
            SelCh_ir = ga.recombin(recombinStyle, P_ir, recopt, SUBPOP)  # 重组
            SelCh_ir = ga.mutbga(SelCh_ir, FieldDR_ir, pm)  # 变异
            if distribute == True and repnum[index] > P_i.shape[0] * 0.01:  # 当最优个体重复率高达1%时，进行一次高斯变异
                SelCh_ir = ga.mutgau(SelCh_ir, FieldDR_ir, pm)  # 高斯变异

            SelCh[:, index_ib] = variable
            SelCh[:, index_ir] = SelCh_ir
        else:
            SelCh = ga.recombin(recombinStyle, P_i, recopt, SUBPOP)  # 重组
            SelCh = ga.mutbga(SelCh, FieldDR_i, pm)  # 变异
            if distribute == True and repnum[index] > P_i.shape[0] * 0.01:  # 当最优个体重复率高达1%时，进行一次高斯变异
                SelCh = ga.mutgau(SelCh, FieldDR_i, pm)  # 高斯变异
        return SelCh

    def is2(self, FieldD):
        r = FieldD[2, :] - FieldD[1, :]
        result = [self.dec2bin(x, l) for x, l in zip(r, FieldD[0, :])]
        if (np.array(result) == 1).all():
            return True
        else:
            return False

    def rv2bs(self, gen, FieldD):
        result = []
        for individual in gen:
            gen_i = []
            for g, u, c, l in zip(individual, FieldD[1,:], FieldD[3, :], FieldD[0, :]):
                g_b = self.dec2bin(g-u, l)
                if c == 1:
                    g_g =self.bin2gary(g_b)
                    gen_i.extend(g_g)
                elif c == 0:
                    gen_i.extend(g_b)
            result.append(gen_i)
        return np.array(result)

    def initilize_B(self):
        B = []
        for SubCom_i in self.SubCom:
            FieldDR_i = self.FieldDR[:, SubCom_i]
            B_i = ga.crtrp(1, FieldDR_i)
            B_i = self.b_coding(B_i, SubCom_i)
            B.extend(list(B_i[0]))
        return np.array(B).reshape(1,-1)

    def initialize_offspring(self, NIND, B):
        P = []
        ObjV = []
        LegV = []
        for SubCom_i in self.SubCom:
            FieldDR_i = self.FieldDR[:, SubCom_i]
            # 生成初始种群
            P_i = ga.crtrp(NIND, FieldDR_i)
            # 进行必要的二进制编码
            P_i = self.b_coding(P_i, SubCom_i)
            # 替换context vector中个体基因
            Chrom = B.copy().repeat(NIND, axis=0)
            Chrom[:, SubCom_i] = P_i
            LegV_i = np.ones((NIND, 1))
            # 求子问题的目标函数值
            [ObjV_i, LegV_i] = self.aimfunc(Chrom, LegV_i)
            # update the BO model
            self.surrogate.update_model()
            # 各个种群的初始基因
            P.append(P_i)
            # 各个种群的初始函数值
            ObjV.append(ObjV_i)
            # 生成可行性列向量，元素为1表示对应个体是可行解，0表示非可行解
            LegV.append(LegV_i)
        return P, ObjV, LegV

    def update_context_vector(self,maxormin, Chrom_, B, F_B, ObjVSel_, LegVSel_):
        _B, _F_B= B, F_B
        for j, (ObjVSel_j, LegVSel_j) in enumerate(zip(ObjVSel_, LegVSel_)):
            if maxormin == 1:
                if ObjVSel_j < F_B and LegVSel_j == 1:
                    _F_B = ObjVSel_j
                    _B[0] = Chrom_[j, :]
            if maxormin == -1:
                if ObjVSel_j > F_B and LegVSel_j == 1:
                    _F_B = ObjVSel_j
                    _B[0] = Chrom_[j, :]
        return _B, _F_B

    def draw(self, NVAR, pop_trace, var_trace, best_ObjV, best_gen, times):
        ga.trcplot(pop_trace, [['种群最优个体目标函数值']])
        print('最优的目标函数值为：%s' % (best_ObjV))
        print('最优的控制变量值为：')
        for i in range(NVAR):
            print(var_trace[best_gen, i])
        print('有效进化代数：%s' % (pop_trace.shape[0]))
        print('最优的一代是第 %s 代' % (best_gen + 1))
        print('时间已过 %s 秒' % (times))

    def deal_records(self, maxormin, pop_trace, var_trace):
        delIdx = np.where(np.isnan(pop_trace))[0]
        _pop_trace = np.delete(pop_trace, delIdx, 0)
        _var_trace = np.delete(var_trace, delIdx, 0)
        if pop_trace.shape[0] == 0:
            raise RuntimeError('error: no feasible solution. (有效进化代数为0，没找到可行解。)')
        if maxormin == 1:
            best_gen = np.argmin(pop_trace[:, 0])  # 记录最优种群是在哪一代
            best_ObjV = np.min(pop_trace[:, 0])
        elif maxormin == -1:
            best_gen = np.argmax(pop_trace[:, 0])  # 记录最优种群是在哪一代
            best_ObjV = np.max(pop_trace[:, 0])
        return _pop_trace, _var_trace, best_gen, best_ObjV

    def add_distribute(self,ObjV_i,FitnV):
        idx = np.argsort(ObjV_i[:, 0], 0)
        dis = np.diff(ObjV_i[idx, 0]) / (
                np.max(ObjV_i[idx, 0]) - np.min(ObjV_i[idx, 0]) + 1)  # 差分计算距离的修正偏移量
        dis = np.hstack([dis, dis[-1]])
        dis = dis + np.min(dis)  # 修正偏移量+最小量=修正绝对量
        FitnV[idx, 0] *= np.exp(dis)  # 根据相邻距离修改适应度，突出相邻距离大的个体，以增加种群的多样性

    def coe_surrogate(self, recopt=0.9, pm=0.1, MAXGEN=100, NIND=10, init_points = 50,
                      maxormin=1, SUBPOP=1, GGAP=0.5, online = True, eva = 1, interval=1,
                      selectStyle='sus', recombinStyle='xovdp', distribute = False, LHS_path = None, drawing= False):

        # ==========================初始化配置===========================
        # 得到控制变量的个数
        NVAR = self.FieldDR.shape[1]
        # 定义进化记录器，初始值为nan
        pop_trace = (np.zeros((MAXGEN, 1)) * np.nan)
        # 定义变量记录器，记录控制变量值，初始值为nan
        var_trace = (np.zeros((MAXGEN, NVAR)) * np.nan)
        # 初始化重复个体数为0
        repnum = [0] * len(self.SubCom)
        # 初始化代理模型
        self.surrogate.initial_model(init_points = init_points, LHS_path = LHS_path, is_LHS = True, lazy = False)
        # 定义初始context vector
        B = self.initilize_B()
        # 求初代context vector 的 fitness
        [F_B, LegV_B] = self.aimfunc(B, np.ones((1, 1)))
        # 初始化各个子种群
        P, ObjV, LegV = self.initialize_offspring(NIND,B)
        # 初始代数
        gen = 0
        # 用于记录在“遗忘策略下”被忽略的代数
        badCounter = 0

        # =========================开始遗传算法进化=======================
        # 开始进化！！
        # 开始计时
        start_time = time.time()
        # 设置一个用原函数评估的代数间隔
        estimation = interval - 1
        while gen < MAXGEN:
            # 若多花了10倍的迭代次数仍没有可行解出现，则跳出
            if badCounter >= 10 * MAXGEN:
                break
            # 本轮估计次数+1
            estimation += 1
            # 每一轮进化轮流进化各个子种群
            for index, (SubCom_i, P_i, ObjV_i, LegV_i) in enumerate(zip(self.SubCom, P, ObjV, LegV)):
                # 进行遗传算子的重组和变异，生成子代
                SelCh = self.remu(P_i, SubCom_i, recombinStyle, recopt, SUBPOP, pm, distribute, repnum, index)
                # 替换context vector中个体基因
                Chrom = B.copy().repeat(NIND, axis=0)
                Chrom[:, SubCom_i] = SelCh
                # 初始化育种种群的可行性列向量
                LegVSel = np.ones((Chrom.shape[0], 1))
                # get the estimated value thought the surrogate
                ObjVSel = self.surrogate.predict(Chrom).reshape(-1, 1)
                # 估计子种群的acquisition function value
                guess = self.surrogate.guess(Chrom)
                # 如果评估次数大于代数间隔就进行原函数评估
                if estimation >= interval:
                    # 找到估计最好的eva个基因序号
                    best_guess = guess.argsort()[0:int(eva)]
                    # 找到估计最好的eva个基因
                    Chrom_ = np.array(Chrom)[best_guess]
                    # 初始化实际评估种群的可行性列向量
                    LegVSel_ = np.ones((Chrom_.shape[0], 1))
                    # 求育种种群的目标函数值
                    [ObjVSel_, LegVSel_] = self.aimfunc(Chrom_, LegVSel_)
                    # 如果在线更新，则更新代理模型
                    if online:
                        # update the BO model
                        self.surrogate.update_model()
                    # replace the estimated value by real value
                    ObjVSel[best_guess] = ObjVSel_
                    LegVSel[best_guess] = LegVSel_
                    # 更新context vector 及其fitness （已经考虑排除不可行解）
                    B, F_B = self.update_context_vector(maxormin, Chrom_, B, F_B, ObjVSel_, LegVSel_)

                # 父子合并
                P_i = np.vstack([P_i, SelCh])
                ObjV_i = np.vstack([ObjV_i, ObjVSel])
                LegV_i = np.vstack([LegV_i, LegVSel])
                # 对合并的种群进行适应度评价
                FitnV = ga.ranking(maxormin * ObjV_i, LegV_i, None, SUBPOP)
                # 调用罚函数
                FitnV = self.punfunc(LegV_i, FitnV)
                # 获取最优个体的下标
                bestIdx = np.argmax(FitnV)
                # 排除非可行解
                if LegV_i[bestIdx] != 0:
                    # feasible = np.where(LegV_i != 0)[0]
                    # 计算最优个体重复数
                    repnum[index] = len(np.where(ObjV_i[bestIdx] == ObjV_i)[0])
                    # badCounter计数器清零
                    badCounter = 0
                else:
                    # 忽略这一代
                    gen -= 1
                    badCounter += 1
                # 若要增强种群的分布性（可能会造成收敛慢）
                if distribute == True:
                    self.add_distribute(ObjV_i, FitnV)
                # 选择个体生成新一代种群
                [P_i, ObjV_i, LegV_i] = ga.selecting(selectStyle, P_i, FitnV, GGAP, SUBPOP, ObjV_i,
                                                     LegV_i)
                # 将子种群情况更新到总种群中去
                P[index], ObjV[index], LegV[index] = P_i, ObjV_i, LegV_i

            # 如果估计次数大于间隔则清零计数
            if estimation >= interval:
                estimation = 0
            # 记录当代目标函数的最优值
            pop_trace[gen, 0] = F_B
            # 记录当代最优的控制变量值
            var_trace[gen, :] = B[0]
            # 增加代数
            gen += 1
        # 结束计时
        end_time = time.time()
        times = end_time - start_time

        # ====================处理进化记录==================================
        # 处理进化记录并获取最佳结果
        pop_trace, var_trace, best_gen, best_ObjV = self.deal_records(maxormin, pop_trace, var_trace)
        # 绘图&输出结果
        if drawing:
            self.draw(NVAR, pop_trace, var_trace, best_ObjV, best_gen, times)
        # 返回最优个体及目标函数值
        return best_gen, best_ObjV

    def coe(self, recopt=0.9, pm=0.1, MAXGEN=100, NIND=10,
            maxormin=1, SUBPOP=1, GGAP=0.5,
            selectStyle='sus', recombinStyle='xovdp', distribute = False, drawing= False):

        # ==========================初始化配置===========================
        # 得到控制变量的个数
        NVAR = self.FieldDR.shape[1]
        # 定义进化记录器，初始值为nan
        pop_trace = (np.zeros((MAXGEN, 1)) * np.nan)
        # 定义变量记录器，记录控制变量值，初始值为nan
        var_trace = (np.zeros((MAXGEN, NVAR)) * np.nan)
        # 初始化重复个体数为0
        repnum = [0] * len(self.SubCom)
        # 定义初始context vector
        B = self.initilize_B()
        # 求初代context vector 的 fitness
        [F_B, LegV_B] = self.aimfunc(B, np.ones((1, 1)))
        # 初始化各个子种群
        P, ObjV, LegV = self.initialize_offspring(NIND,B)
        # 初始代数
        gen = 0
        # 用于记录在“遗忘策略下”被忽略的代数
        badCounter = 0

        # =========================开始遗传算法进化=======================
        # 开始进化！！
        # 开始计时
        start_time = time.time()
        while gen < MAXGEN:
            # 若多花了10倍的迭代次数仍没有可行解出现，则跳出
            if badCounter >= 10 * MAXGEN:
                break
            # 每一轮进化轮流进化各个子种群
            for index, (SubCom_i, P_i, ObjV_i, LegV_i) in enumerate(zip(self.SubCom, P, ObjV, LegV)):
                # 进行遗传算子的重组和变异，生成子代
                SelCh = self.remu(P_i, SubCom_i, recombinStyle, recopt, SUBPOP, pm, distribute, repnum, index)
                # 替换context vector中个体基因
                Chrom = B.copy().repeat(NIND, axis=0)
                Chrom[:, SubCom_i] = SelCh
                # 初始化育种种群的可行性列向量
                LegVSel = np.ones((Chrom.shape[0], 1))
                # get the estimated value thought the surrogate
                ObjVSel = self.surrogate.predict(Chrom).reshape(-1, 1)
                # 求育种种群的目标函数值
                [ObjVSel, LegVSel] = self.aimfunc(Chrom, LegVSel)
                # 更新context vector 及其fitness （已经考虑排除不可行解）
                B, F_B = self.update_context_vector(maxormin, Chrom, B, F_B, ObjVSel, LegVSel)
                # 父子合并
                P_i = np.vstack([P_i, SelCh])
                ObjV_i = np.vstack([ObjV_i, ObjVSel])
                LegV_i = np.vstack([LegV_i, LegVSel])
                # 对合并的种群进行适应度评价
                FitnV = ga.ranking(maxormin * ObjV_i, LegV_i, None, SUBPOP)
                # 调用罚函数
                FitnV = self.punfunc(LegV_i, FitnV)
                # 获取最优个体的下标
                bestIdx = np.argmax(FitnV)
                # 排除非可行解
                if LegV_i[bestIdx] != 0:
                    # feasible = np.where(LegV_i != 0)[0]
                    # 计算最优个体重复数
                    repnum[index] = len(np.where(ObjV_i[bestIdx] == ObjV_i)[0])
                    # badCounter计数器清零
                    badCounter = 0
                else:
                    # 忽略这一代
                    gen -= 1
                    badCounter += 1
                # 若要增强种群的分布性（可能会造成收敛慢）
                if distribute == True:
                    self.add_distribute(ObjV_i, FitnV)
                # 选择个体生成新一代种群
                [P_i, ObjV_i, LegV_i] = ga.selecting(selectStyle, P_i, FitnV, GGAP, SUBPOP, ObjV_i,
                                                     LegV_i)
                # 将子种群情况更新到总种群中去
                P[index], ObjV[index], LegV[index] = P_i, ObjV_i, LegV_i

            # 记录当代目标函数的最优值
            pop_trace[gen, 0] = F_B
            # 记录当代最优的控制变量值
            var_trace[gen, :] = B[0]
            # 增加代数
            gen += 1
        # 结束计时
        end_time = time.time()
        times = end_time - start_time

        # ====================处理进化记录==================================
        # 处理进化记录并获取最佳结果
        pop_trace, var_trace, best_gen, best_ObjV = self.deal_records(maxormin, pop_trace, var_trace)
        # 绘图&输出结果
        if drawing:
            self.draw(NVAR, pop_trace, var_trace, best_ObjV, best_gen, times)
        # 返回最优个体及目标函数值
        return best_gen, best_ObjV


if __name__ == "__main__":
    def rosen(alpha=1e2, **X):
        x = [X[key]/10 for key in X.keys()]
        """Rosenbrock test objective function"""
        x = [x] if np.isscalar(x[0]) else x  # scalar into list
        x = np.asarray(x)
        f = [sum(alpha * (x[:-1] ** 2 - x[1:]) ** 2 + (1. - x[:-1]) ** 2) for x in x]
        return f if len(f) > 1 else f[0]  # 1-element-list into scalar

    dim = 8
    keys = ['x', 'y', 'z', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2']
    ranges = np.vstack([[0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 9], [0, 10], [0, 9]]).T
    borders = np.vstack([[0, 0]] * dim).T
    precisions = np.array([4, 4, 4, 4, 4, 0, 4, 0])
    codes = np.array([None, None, None, None, None, 1, None, 1])
    scales = np.array([0] * dim)
    FieldDR = ga.crtfld(ranges, borders, list(precisions))
    SubCom = np.array([[0, 1], [2, 3], [4, 5, 6, 7]])
    radom_state = 1

    coe = CoE_surrogate_mixgentype(rosen, None, SubCom, ranges, borders, precisions, codes, scales, keys, radom_state,
                                   surrogate_type = 'gp',
                                   acq='ucb', kappa=2.576, xi=0.0)
    best_gen, best_ObjV = coe.coe_surrogate(recopt=0.9, pm=0.1, MAXGEN=100, NIND=10,
                                            init_points=50,
                                            maxormin=1, SUBPOP=1, GGAP=0.5, online=False, eva=1,
                                            interval=1,
                                            selectStyle='sus', recombinStyle='xovdp',
                                            distribute=False, drawing=False)


