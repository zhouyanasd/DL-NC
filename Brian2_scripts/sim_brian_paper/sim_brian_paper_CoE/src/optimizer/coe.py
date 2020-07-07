# -*- coding: utf-8 -*-
"""
    The optimization methods used for NAS.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""
from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.optimizer.bayesian import BayesianOptimization

import time

import numpy as np
import geatpy as ga


class CoE_surrgate():
    def __init__(self, f, f_p, SubCom, ranges, borders, precisions, keys, acq, kappa=2.576, xi=0.0,
                 opt='cma', **gp_params):
        self.f = f # for BO with dict input
        self.f_p = f_p
        self.SubCom = SubCom
        self.FieldDR = ga.crtfld(ranges, borders, list(precisions))
        self.keys = keys
        self.surrogate = BayesianOptimization(
            f=f,
            pbounds= dict(zip(self.keys, [tuple(x) for x in self.FieldDR.T])), # 此处需要修改到和borders匹配的形式
            random_state=1,
            acq=acq, opt=opt, kappa = kappa, xi =xi, **gp_params
        )

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
                                   selectStyle='sus', recombinStyle='xovdp', distribute=True, drawing=0):

        """==========================初始化配置==========================="""
        # GGAP = 0.5  # 因为父子合并后选择，因此要将代沟设为0.5以维持种群规模
        NVAR = self.FieldDR.shape[1]  # 得到控制变量的个数
        # 定义进化记录器，初始值为nan
        pop_trace = (np.zeros((MAXGEN, 1)) * np.nan)
        # 定义变量记录器，记录控制变量值，初始值为nan
        var_trace = (np.zeros((MAXGEN, NVAR)) * np.nan)
        repnum = [0] * len(self.SubCom)  # 初始化重复个体数为0
        self.surrogate.initial_model(init_points = init_points, LHS_path = None, is_LHS = True, lazy = False)
        ax = None  # 存储上一帧图形
        """=========================开始遗传算法进化======================="""
        if problem == 'R':
            B = ga.crtrp(1, self.FieldDR)  # 定义初始contex vector
        elif problem == 'I':
            B = ga.crtip(1, self.FieldDR)
        [F_B, LegV_B] = self.aimfunc(B, np.ones((1, 1)))  # 求初代contex vector 的 fitness

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
            [ObjV_i, LegV_i] = self.aimfunc(Chrom, LegV_i)  # 求子问题的目标函数值
            self.surrogate.update_model()  # update the BO model
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
                ObjVSel = self.surrogate._gp.predict(Chrom).reshape(-1, 1)  # get the estimated value

                guess = self.surrogate.guess_fixedpoint(Chrom)  # 估计子种群的acquisition function value

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


class Coe_surrogate_mixgentype(CoE_surrgate):
    '''
    codes和scales中不需要编码的部分用None来代替。
    '''
    def __init__(self,f, f_p, SubCom, ranges, borders, precisions, codes, scales, keys, acq, kappa=2.576, xi=0.0,
                     opt='cma', **gp_params):
        super().__init__(f, f_p, SubCom, ranges, borders, precisions, keys, acq, kappa=kappa, xi=xi,
                     opt=opt, **gp_params)
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

    def bin2Dec(self, binary):
        result = 0
        for i in range(len(binary)):
            result += int(binary[-(i + 1)]) * pow(2, i)
        return result

    def gray2bin(self, gray):
        result = []
        result.append(gray[0])
        for i, g in enumerate(gray[1:]):
            result.append(g ^ result[i])
        return result

    def dec2bin(self,num, l):
        result = []
        if num < 0:
            return '-' + dec2bin(abs(num))
        while True:
            num, remainder = divmod(num, 2)
            result.append(int(remainder))
            if num == 0:
                break
        if len(result) < l:
            result.extend([0] * (l - len(result)))
        return result[::-1]

    def bin2gary(self, binary):
        result = []
        result.append(binary[0])
        for i, b in enumerate(binary[1:]):
            result.append(b ^ binary[i])
        return result

    def rv2bs(self, gen, FieldD):
        result = []
        for individual in gen:
            gen_i = []
            for g, c, l in zip(individual, FieldD[3, :], FieldD[0, :]):
                g_b = self.dec2bin(g, l)
                if c == 1:
                    g_g =self.bin2gary(g_b)
                    gen_i.extend(g_g)
                elif c == 0:
                    gen_i.extend(g_b)
            result.append(gen_i)
        return np.array(result)

    def coe_surrogate_real_templet(self, recopt=0.9, pm=0.1, MAXGEN=100, NIND=10, init_points = 50,
                                   problem='R', maxormin=1, SUBPOP=1, GGAP=0.5, online = True, eva = 1, interval=1,
                                   selectStyle='sus', recombinStyle='xovdp', distribute=True, drawing=0):

        """==========================初始化配置==========================="""
        NVAR = self.FieldDR.shape[1]  # 得到控制变量的个数
        # 定义进化记录器，初始值为nan
        pop_trace = (np.zeros((MAXGEN, 1)) * np.nan)
        # 定义变量记录器，记录控制变量值，初始值为nan
        var_trace = (np.zeros((MAXGEN, NVAR)) * np.nan)
        repnum = [0] * len(self.SubCom)  # 初始化重复个体数为0
        self.surrogate.initial_model(init_points = init_points, LHS_path = None, is_LHS = True, lazy = False)
        ax = None  # 存储上一帧图形

        """=========================开始遗传算法进化======================="""
        B = ga.crtrp(1, self.FieldDR)  # 定义初始contex vector
        [F_B, LegV_B] = self.aimfunc(B, np.ones((1, 1)))  # 求初代contex vector 的 fitness

        # 初始化各个子种群
        P = []
        ObjV = []
        LegV = []
        for SubCom_i in self.SubCom:
            FieldDR_i = self.FieldDR[:, SubCom_i]

            P_i = ga.crtrp(NIND, FieldDR_i)  # 生成初始种群

            P_i = self.b_coding(P_i, SubCom_i)

            Chrom = B.copy().repeat(NIND, axis=0)  # 替换contex vector中个体基因
            Chrom[:, SubCom_i] = P_i
            LegV_i = np.ones((NIND, 1))
            [ObjV_i, LegV_i] = self.aimfunc(Chrom, LegV_i)  # 求子问题的目标函数值
            self.surrogate.update_model()  # update the BO model
            P.append(P_i)
            ObjV.append(ObjV_i)
            LegV.append(LegV_i)  # 生成可行性列向量，元素为1表示对应个体是可行解，0表示非可行解
        gen = 0
        badCounter = 0  # 用于记录在“遗忘策略下”被忽略的代数

        # 开始进化！！
        start_time = time.time()  # 开始计时
        estimation = interval - 1  # counter and make sure the first time could be evaluated
        while gen < MAXGEN:
            if badCounter >= 10 * MAXGEN:  # 若多花了10倍的迭代次数仍没有可行解出现，则跳出
                break
            estimation += 1
            for index, (SubCom_i, P_i, ObjV_i, LegV_i) in enumerate(zip(self.SubCom, P, ObjV, LegV)):
                # 进行遗传算子，生成子代

                SelCh = self.remu(P_i, SubCom_i,recombinStyle,recopt,SUBPOP,pm,distribute,index)

                Chrom = B.copy().repeat(NIND, axis=0)  # 替换contex vector中个体基因
                Chrom[:, SubCom_i] = SelCh
                LegVSel = np.ones((Chrom.shape[0], 1))  # 初始化育种种群的可行性列向量
                ObjVSel = self.surrogate._gp.predict(Chrom).reshape(-1, 1)  # get the estimated value

                guess = self.surrogate.guess_fixedpoint(Chrom)  # 估计子种群的acquisition function value

                if estimation >= interval:  # 设置一个更新间隔
                    best_guess = guess.argsort()[0:int(eva)]  # 找到估计最好的eva个基因序号

                    Chrom_ = np.array(Chrom)[best_guess]  # 找到估计最好的eva个基因
                    LegVSel_ = np.ones((Chrom_.shape[0], 1))  # 初始化实际评估种群的可行性列向量
                    [ObjVSel_, LegVSel_] = self.aimfunc(Chrom_, LegVSel_)  # 求育种种群的目标函数值
                    if online:
                        self.surrogate.update_model()  # update the BO model

                    ObjVSel[best_guess] = ObjVSel_  # replace the estimated value by real value
                    LegVSel[best_guess] = LegVSel_

                    # 更新context vector 及其fitness （已经考虑排除不可行解）
                    for j, (ObjVSel_j, LegVSel_j) in enumerate(zip(ObjVSel_, LegVSel_)):
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
            gen += 1
        end_time = time.time()  # 结束计时
        times = end_time - start_time

        # ====================后处理进化记录器==================================
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


