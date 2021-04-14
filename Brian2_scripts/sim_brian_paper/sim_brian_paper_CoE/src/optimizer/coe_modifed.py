# -*- coding: utf-8 -*-
"""
    The optimization methods used for NAS.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""
from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.core import BaseFunctions
from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.optimizer.surrogate import TargetSpace, create_surrogate
from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.optimizer import ga as ga

import time, pickle

import numpy as np


class OptimizerBase(BaseFunctions):
    def __init__(self, f, f_p, SubCom, ranges, borders, precisions, codes, scales, keys, random_state, maxormin):
        super().__init__()
        self.f = f  # for BO with dict input
        self.f_p = f_p
        self.SubCom = SubCom
        self.keys = keys
        self.ranges = ranges
        self.borders = borders
        self.precisions = precisions
        self.codes = codes
        self.scales = scales
        self.maxormin = maxormin
        self.FieldDR = ga.crtfld(ranges, borders, list(precisions))
        self._space = TargetSpace(f, keys, ranges, borders, precisions, random_state)

        self.gen = 0
        self.start_time = time.time()
        self.end_time = time.time()

        self.NVAR = self.FieldDR.shape[1]
        self.pop_trace = np.array([]).reshape(-1, 1)
        self.var_trace = np.array([]).reshape(-1, self.NVAR)

        np.random.set_state(random_state)

    def set_random_state_by_time(self):
        np.random.set_state(self.start_time)

    def get_times(self):
        return self.end_time - self.start_time

    def aimfunc(self, Phen, LegV):  # for GA with the LegV input and output
        res = []
        for phen in Phen:
            res.append(self._space.probe(phen))  # probe replace f and use space checking and register
        return [np.array(res).reshape(-1, 1), LegV]

    def punfunc(self, LegV, FitnV):
        if self.f_p == None:
            return FitnV
        else:
            return self.f_p(LegV, FitnV)

    def save_states(self, B, F_B, ObjV, LegV, repnum, pop_trace, var_trace, P, gen, times, numpy_state):
        with open('./coe.p', 'wb') as f:
            pickle.dump((B, F_B, ObjV, LegV, repnum, pop_trace, var_trace, P, gen, times, numpy_state),
                        f, pickle.HIGHEST_PROTOCOL)

    def load_states(self):
        with open('./coe.p', 'rb') as f:
            B, F_B, ObjV, LegV, repnum, pop_trace, var_trace, P, gen, times, numpy_state =  pickle.load(f)
        return B, F_B, ObjV, LegV, repnum, pop_trace, var_trace, P, gen, times, numpy_state

    def deal_records(self):
        delIdx = np.where(np.isnan(self.pop_trace))[0]
        self.pop_trace = np.delete(self.pop_trace, delIdx, 0)
        self.pop_trace = np.delete(self.var_trace, delIdx, 0)
        if self.pop_trace.shape[0] == 0:
            raise RuntimeError('error: no feasible solution. (有效进化代数为0，没找到可行解。)')
        if self.maxormin == 1:
            self.best_gen = np.argmin(self.pop_trace[:, 0])  # 记录最优种群是在哪一代
            self.best_ObjV = np.min(self.pop_trace[:, 0])
        elif self.maxormin == -1:
            self.best_gen = np.argmax(self.pop_trace[:, 0])  # 记录最优种群是在哪一代
            self.best_ObjV = np.max(self.pop_trace[:, 0])

    def draw(self):
        ga.trcplot(self.pop_trace, [['种群最优个体目标函数值']])
        print('最优的目标函数值为：%s' % (self.best_ObjV))
        print('最优的控制变量值为：')
        for i in range(self.NVAR):
            print(self.var_trace[self.best_gen, i])
        print('有效进化代数：%s' % (self.pop_trace.shape[0]))
        print('最优的一代是第 %s 代' % (self.best_gen + 1))
        print('时间已过 %s 秒' % (self.get_times()))


class CoE(OptimizerBase):
    def __init__(self, f, f_p, SubCom, ranges, borders, precisions, codes, scales, keys, random_state, maxormin,
                 load_continue):
        super().__init__(f, f_p, SubCom, ranges, borders, precisions, codes, scales, keys, random_state, maxormin)

        if load_continue:
            self.reload()
        else:
            # 初始化重复个体数为0
            self.repnum = [0] * len(self.SubCom)
            # 定义初始context vector
            self.B = self.initilize_B()
            # 求初代context vector 的 fitness
            [self.F_B, self.LegV_B] = self.aimfunc(self.B, np.ones((1, 1)))
            # 初始化各个子种群
            self.P = []
            self.ObjV = []
            self.LegV = []
            for SubCom_i in self.SubCom:
                FieldDR_i = self.FieldDR[:, SubCom_i]
                # 生成初始种群
                P_i = ga.crtrp(self.NIND, FieldDR_i)
                # 进行必要的二进制编码
                P_i = self.b_coding(P_i, SubCom_i)
                # 替换context vector中个体基因
                Chrom = self.B.copy().repeat(self.NIND, axis=0)
                Chrom[:, SubCom_i] = P_i
                Chrom = self._space.add_precision(Chrom, self._space._precisions)
                LegV_i = np.ones((self.NIND, 1))
                # 求子问题的目标函数值
                [ObjV_i, LegV_i] = self.aimfunc(Chrom, LegV_i)
                # 各个种群的初始基因
                self.P.append(P_i)
                # 各个种群的初始函数值
                self.ObjV.append(ObjV_i)
                # 生成可行性列向量，元素为1表示对应个体是可行解，0表示非可行解
                self.LegV.append(LegV_i)
            # 根据时间改变随机数
            self.set_random_state_by_time()

    def reload(self):
        # 重新载入历史的进化数据
        self.B, self.F_B, self.ObjV, self.LegV, self.repnum, self.pop_trace, self.var_trace, \
        self.P, self.gen, times, numpy_state = self.load_states()
        # 初始化计时
        self.start_time = time.time() - times
        self.end_time = time.time()
        # 恢复随机数
        np.random.set_state(numpy_state)

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
            if not ga.is2(FieldD):
                raise Exception('worng range of binary coding')
            Lind = np.sum(FieldD[0, :])  # 种群染色体长度
            P_ib = ga.crtbp(NIND, Lind)  # 生成初始种
            variable = ga.bs2rv(P_ib, FieldD)  # 解码
            P_i[:, index_ib] = variable
            return P_i
        else:
            return P_i

    def initilize_B(self):
        B = []
        for SubCom_i in self.SubCom:
            FieldDR_i = self.FieldDR[:, SubCom_i]
            B_i = ga.crtrp(1, FieldDR_i)
            B_i = self.b_coding(B_i, SubCom_i)
            B.extend(list(B_i[0]))
        return self._space.add_precision(np.array(B).reshape(1,-1), self._space._precisions)

    def update_context_vector(self, Chrom_, B, F_B, ObjVSel_, LegVSel_):
        _B, _F_B= B, F_B
        for j, (ObjVSel_j, LegVSel_j) in enumerate(zip(ObjVSel_, LegVSel_)):
            if self.maxormin == 1:
                if ObjVSel_j < F_B and LegVSel_j == 1:
                    _F_B = ObjVSel_j
                    _B[0] = Chrom_[j, :]
            if self.maxormin == -1:
                if ObjVSel_j > F_B and LegVSel_j == 1:
                    _F_B = ObjVSel_j
                    _B[0] = Chrom_[j, :]
        return _B, _F_B

    def add_distribute(self,ObjV_i,FitnV):
        idx = np.argsort(ObjV_i[:, 0], 0)
        dis = np.diff(ObjV_i[idx, 0]) / (
                np.max(ObjV_i[idx, 0]) - np.min(ObjV_i[idx, 0]) + 1)  # 差分计算距离的修正偏移量
        dis = np.hstack([dis, dis[-1]])
        dis = dis + np.min(dis)  # 修正偏移量+最小量=修正绝对量
        FitnV[idx, 0] *= np.exp(dis)  # 根据相邻距离修改适应度，突出相邻距离大的个体，以增加种群的多样性

    def update_generation(self):
        # 记录当代目标函数的最优值
        self.np_append(self.pop_trace, self.F_B)
        # 记录当代最优的控制变量值
        self.np_append(self.var_trace, self.B[0])
        # 增加代数
        self.gen += 1
        # 更新计时
        self.end_time = time.time()
        self.save_states(self.B, self.F_B, self.ObjV, self.LegV, self.repnum, self.pop_trace, self.var_trace,
                         self.P, self.gen, self.get_times(), np.random.get_state())

    def optimize(self, recopt=0.9, pm=0.1, MAXGEN=100, NIND=10, SUBPOP=1, GGAP=0.5,
                 selectStyle='sus', recombinStyle='xovdp', distribute = False):
        # 用于记录在“遗忘策略下”被忽略的代数
        badCounter = 0

        # =========================开始遗传算法进化=======================
        # 开始进化！！
        while self.gen < MAXGEN:
            # 若多花了10倍的迭代次数仍没有可行解出现，则跳出
            if badCounter >= 10 * MAXGEN:
                break
            # 每一轮进化轮流进化各个子种群
            for index, (SubCom_i, P_i, ObjV_i, LegV_i) in enumerate(zip(self.SubCom, self.P, self.ObjV, self.LegV)):
                # 进行遗传算子的重组和变异，生成子代
                SelCh = self.remu(P_i, SubCom_i, recombinStyle, recopt, SUBPOP, pm, distribute, self.repnum, index)
                # 替换context vector中个体基因
                Chrom = self.B.copy().repeat(NIND, axis=0)
                Chrom[:, SubCom_i] = SelCh
                # 初始化育种种群的可行性列向量
                LegVSel = np.ones((Chrom.shape[0], 1))
                # 求育种种群的目标函数值
                [ObjVSel, LegVSel] = self.aimfunc(Chrom, LegVSel)
                # 更新context vector 及其fitness （已经考虑排除不可行解）
                self.B, self.F_B = self.update_context_vector(Chrom, self.B, self.F_B, ObjVSel, LegVSel)
                # 父子合并
                P_i = np.vstack([P_i, SelCh])
                ObjV_i = np.vstack([ObjV_i, ObjVSel])
                LegV_i = np.vstack([LegV_i, LegVSel])
                # 对合并的种群进行适应度评价
                FitnV = ga.ranking(self.maxormin * ObjV_i, LegV_i, None, SUBPOP)
                # 调用罚函数
                FitnV = self.punfunc(LegV_i, FitnV)
                # 获取最优个体的下标
                bestIdx = np.argmax(FitnV)
                # 排除非可行解
                if LegV_i[bestIdx] != 0:
                    # feasible = np.where(LegV_i != 0)[0]
                    # 计算最优个体重复数
                    self.repnum[index] = len(np.where(ObjV_i[bestIdx] == ObjV_i)[0])
                    # badCounter计数器清零
                    badCounter = 0
                else:
                    # 忽略这一代
                    self.gen -= 1
                    badCounter += 1
                # 若要增强种群的分布性（可能会造成收敛慢）
                if distribute == True:
                    self.add_distribute(ObjV_i, FitnV)
                # 选择个体生成新一代种群
                [P_i, ObjV_i, LegV_i] = ga.selecting(selectStyle, P_i, FitnV, GGAP, SUBPOP, ObjV_i,
                                                     LegV_i)
                # 将子种群情况更新到总种群中去
                self.P[index], self.ObjV[index], self.LegV[index] = P_i, ObjV_i, LegV_i

            self.update_generation()
        # ====================处理进化记录==================================
        # 处理进化记录并获取最佳结果
        self.deal_records(self.pop_trace, self.var_trace)



class CoE_surrogate(CoE):
    def __init__(self, f, f_p, SubCom, ranges, borders, precisions, codes, scales, keys, random_state, maxormin,
                 surrogate_type, init_points, LHS_path, load_continue, **surrogate_parameters):
        super().__init__(f, f_p, SubCom, ranges, borders, precisions, codes, scales, keys, random_state, maxormin)

        self.surrogate = create_surrogate(surrogate_type = surrogate_type , f = f, random_state= random_state,
                                          keys=keys, ranges=ranges, borders=borders, precisions=precisions,
                                          **surrogate_parameters)
        self._space = self.surrogate._space

        self.surrogate.initial_model(init_points=init_points, LHS_path=LHS_path, is_LHS=True, lazy=False)

        if load_continue:
            self.reload()
        else:
            # 初始化各个子种群
            self.initialize_offspring()
            # 根据时间改变随机数
            self.set_random_state_by_time()

    def initialize_offspring(self, is_eva = False):
        # 初始化重复个体数为0
        self.repnum = [0] * len(self.SubCom)
        # 从代理模型初始化的数据中找到最好的点
        self.B = np.expand_dims(self._space.params[self._space.target.argmin()], 0)
        self.F_B = self._space.target.min()
        self.P = []
        self.ObjV = []
        self.LegV = []
        for SubCom_i in self.SubCom:
            FieldDR_i = self.FieldDR[:, SubCom_i]
            # 生成初始种群
            P_i = ga.crtrp(self.NIND, FieldDR_i)
            # 进行必要的二进制编码
            P_i = self.b_coding(P_i, SubCom_i)
            # 替换context vector中个体基因
            Chrom = self.B.copy().repeat(self.NIND, axis=0)
            Chrom[:, SubCom_i] = P_i
            Chrom = self._space.add_precision(Chrom,self._space._precisions)
            LegV_i = np.ones((self.NIND, 1))
            if is_eva:
                # 求子问题的目标函数值
                [ObjV_i, LegV_i] = self.aimfunc(Chrom, LegV_i)
                # update the BO model
                self.surrogate.update_model()
            else:
                # 初代中确直接用代理评估出来
                ObjV_i = self.surrogate.predict(Chrom).reshape(-1, 1)
            # 各个种群的初始基因
            self.P.append(P_i)
            # 各个种群的初始函数值
            self.ObjV.append(ObjV_i)
            # 生成可行性列向量，元素为1表示对应个体是可行解，0表示非可行解
            self.LegV.append(LegV_i)

    def optimize(self, recopt=0.9, pm=0.1, MAXGEN=100, NIND=10, SUBPOP=1, GGAP=0.5, online = True, eva = 1, interval=1,
                      selectStyle='sus', recombinStyle='xovdp', distribute = False, ):

        # 设置一个用原函数评估的代数间隔
        estimation = interval - 1
        # 用于记录在“遗忘策略下”被忽略的代数
        badCounter = 0

        # =========================开始遗传算法进化=======================
        # 开始进化！！
        while self.gen < MAXGEN:
            # 若多花了10倍的迭代次数仍没有可行解出现，则跳出
            if badCounter >= 10 * MAXGEN:
                break
            # 本轮估计次数+1
            estimation += 1
            # 每一轮进化轮流进化各个子种群
            for index, (SubCom_i, P_i, ObjV_i, LegV_i) in enumerate(zip(self.SubCom, self.P, self.ObjV, self.LegV)):
                # 进行遗传算子的重组和变异，生成子代
                SelCh = self.remu(P_i, SubCom_i, recombinStyle, recopt, SUBPOP, pm, distribute, self.repnum, index)
                # 替换context vector中个体基因
                Chrom = self.B.copy().repeat(NIND, axis=0)
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
                    self.B, self.F_B = self.update_context_vector(Chrom_, self.B, self.F_B, ObjVSel_, LegVSel_)

                # 父子合并
                P_i = np.vstack([P_i, SelCh])
                ObjV_i = np.vstack([ObjV_i, ObjVSel])
                LegV_i = np.vstack([LegV_i, LegVSel])
                # 对合并的种群进行适应度评价
                FitnV = ga.ranking(self.maxormin * ObjV_i, LegV_i, None, SUBPOP)
                # 调用罚函数
                FitnV = self.punfunc(LegV_i, FitnV)
                # 获取最优个体的下标
                bestIdx = np.argmax(FitnV)
                # 排除非可行解
                if LegV_i[bestIdx] != 0:
                    # feasible = np.where(LegV_i != 0)[0]
                    # 计算最优个体重复数
                    self.repnum[index] = len(np.where(ObjV_i[bestIdx] == ObjV_i)[0])
                    # badCounter计数器清零
                    badCounter = 0
                else:
                    # 忽略这一代
                    self.gen -= 1
                    badCounter += 1
                # 若要增强种群的分布性（可能会造成收敛慢）
                if distribute == True:
                    self.add_distribute(ObjV_i, FitnV)
                # 选择个体生成新一代种群
                [P_i, ObjV_i, LegV_i] = ga.selecting(selectStyle, P_i, FitnV, GGAP, SUBPOP, ObjV_i,
                                                     LegV_i)
                # 将子种群情况更新到总种群中去
                self.P[index], self.ObjV[index], self.LegV[index] = P_i, ObjV_i, LegV_i

            # 如果估计次数大于间隔则清零计数
            if estimation >= interval:
                estimation = 0
            self.update_generation()

        # ====================处理进化记录==================================
        # 处理进化记录并获取最佳结果
        self.deal_records(self.pop_trace, self.var_trace)