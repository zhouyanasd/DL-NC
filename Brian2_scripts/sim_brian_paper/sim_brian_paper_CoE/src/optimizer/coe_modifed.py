# -*- coding: utf-8 -*-
"""
    The optimization methods used for NAS.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""
from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.core import BaseFunctions
from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.optimizer.surrogate import create_surrogate
from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.optimizer import ga as ga

import time, pickle

import numpy as np


class OptimizerBase(BaseFunctions):
    def __init__(self):
        super().__init__()

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

    def save_states(self, B, F_B, ObjV, LegV, repnum, pop_trace, var_trace, P, gen, times, numpy_state):
        with open('./coe.p', 'wb') as f:
            pickle.dump((B, F_B, ObjV, LegV, repnum, pop_trace, var_trace, P, gen, times, numpy_state),
                        f, pickle.HIGHEST_PROTOCOL)

    def load_states(self):
        with open('./coe.p', 'rb') as f:
            B, F_B, ObjV, LegV, repnum, pop_trace, var_trace, P, gen, times, numpy_state =  pickle.load(f)
        return B, F_B, ObjV, LegV, repnum, pop_trace, var_trace, P, gen, times, numpy_state


class CoE(OptimizerBase):
    def __init__(self, f, f_p, SubCom, ranges, borders, precisions, codes, scales, keys, random_state):
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
        self.FieldDR = ga.crtfld(ranges, borders, list(precisions))

    def coe(self, recopt=0.9, pm=0.1, MAXGEN=100, NIND=10,
            maxormin=1, SUBPOP=1, GGAP=0.5,
            selectStyle='sus', recombinStyle='xovdp', distribute = False, drawing= False, load_continue = False):

        # ==========================初始化配置===========================
        # 得到控制变量的个数
        NVAR = self.FieldDR.shape[1]
        # 定义进化记录器，初始值为nan
        pop_trace = (np.zeros((MAXGEN, 1)) * np.nan)
        # 定义变量记录器，记录控制变量值，初始值为nan
        var_trace = (np.zeros((MAXGEN, NVAR)) * np.nan)
        if load_continue:
            # 重新载入历史的进化数据
            B, F_B, ObjV, LegV, repnum, pop_trace_, var_trace_, P, gen, times, numpy_state = self.load_states()
            pop_trace[:gen, :] = pop_trace_[:gen, :]
            var_trace[:gen, :] = var_trace_[:gen, :]
            # 初始化计时
            start_time = time.time()-times
            end_time = time.time()
            # 恢复随机数
            np.random.set_state(numpy_state)
        else:
            # 初始化重复个体数为0
            repnum = [0] * len(self.SubCom)
            # 定义初始context vector
            B = self.initilize_B()
            # 求初代context vector 的 fitness
            [F_B, LegV_B] = self.aimfunc(B, np.ones((1, 1)))
            # 初始化各个子种群
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
                Chrom = self._space.add_precision(Chrom, self._space._precisions)
                LegV_i = np.ones((NIND, 1))
                # 求子问题的目标函数值
                [ObjV_i, LegV_i] = self.aimfunc(Chrom, LegV_i)
                # 各个种群的初始基因
                P.append(P_i)
                # 各个种群的初始函数值
                ObjV.append(ObjV_i)
                # 生成可行性列向量，元素为1表示对应个体是可行解，0表示非可行解
                LegV.append(LegV_i)
            # 初始代数
            gen = 0
            # 初始化计时
            start_time = time.time()
            end_time = time.time()
            times = end_time - start_time
            # 根据时间改变随机数
            np.random.seed(int(start_time))
            # 根据时间改变随机数
            np.random.seed(int(start_time))
        # 用于记录在“遗忘策略下”被忽略的代数
        badCounter = 0

        # =========================开始遗传算法进化=======================
        # 开始进化！！
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
            # 更新计时
            end_time = time.time()
            times = end_time - start_time
            self.save_states(B, F_B, ObjV, LegV, repnum, pop_trace, var_trace, P, gen, times, np.random.get_state())

        # ====================处理进化记录==================================
        # 处理进化记录并获取最佳结果
        pop_trace, var_trace, best_gen, best_ObjV = self.deal_records(maxormin, pop_trace, var_trace)
        # 绘图&输出结果
        if drawing:
            self.draw(NVAR, pop_trace, var_trace, best_ObjV, best_gen, times)
        # 返回最优个体及目标函数值
        return best_gen, best_ObjV





class CoE_surrogate(CoE):
    pass








class CoE_surrogate_mixgentype(CoE_surrogate):
    pass