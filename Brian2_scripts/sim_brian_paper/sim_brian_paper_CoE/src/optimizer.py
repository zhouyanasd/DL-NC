# -*- coding: utf-8 -*-
"""
    The optimization methods used for NAS.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""

import re, time
import warnings

import cma
import geatpy as ga
from bayes_opt.event import Events
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction

import numpy as np
from scipy.stats import norm
from numpy import asarray, zeros, zeros_like, tile, array, argmin, mod
from numpy.random import random, randint, rand, seed as rseed, uniform


class DiffEvol(object):
    class _function_wrapper(object):
        def __init__(self, f, args, kwargs):
            self.f = f
            self.args = args
            self.kwargs = kwargs

        def __call__(self, x):
            return self.f(x, *self.args, **self.kwargs)

    def __init__(self, fun, bounds, npop, f=None, c=None, seed=None, maximize=False, vectorize=False, cbounds=(0.25, 1),
                 fbounds=(0.25, 0.75), pool=None, min_ptp=1e-2, args=[], kwargs={}):
        if seed is not None:
            rseed(seed)

        self.minfun = self._function_wrapper(fun, args, kwargs)
        self.bounds = asarray(bounds)
        self.n_pop = npop
        self.n_par = self.bounds.shape[0]
        self.bl = tile(self.bounds[:, 0], [npop, 1])
        self.bw = tile(self.bounds[:, 1] - self.bounds[:, 0], [npop, 1])
        self.m = -1 if maximize else 1
        self.pool = pool
        self.args = args

        if self.pool is not None:
            self.map = self.pool.map
        else:
            self.map = map

        self.periodic = []
        self.min_ptp = min_ptp

        self.cmin = cbounds[0]
        self.cmax = cbounds[1]
        self.cbounds = cbounds
        self.fbounds = fbounds

        self.seed = seed
        self.f = f
        self.c = c

        self._population = asarray(self.bl + random([self.n_pop, self.n_par]) * self.bw)
        self._fitness = zeros(npop)
        self._minidx = None

        self._trial_pop = zeros_like(self._population)
        self._trial_fit = zeros_like(self._fitness)

        if vectorize:
            self._eval = self._eval_vfun
        else:
            self._eval = self._eval_sfun

    @property
    def population(self):
        """The parameter vector population"""
        return self._population

    @property
    def minimum_value(self):
        """The best-fit value of the optimized function"""
        return self._fitness[self._minidx]

    @property
    def minimum_location(self):
        """The best-fit solution"""
        return self._population[self._minidx, :]

    @property
    def minimum_index(self):
        """Index of the best-fit solution"""
        return self._minidx

    def optimize(self, ngen):
        """Run the optimizer for ``ngen`` generations"""
        res = 0
        for res in self(ngen):
            pass
        return res

    def __call__(self, ngen=1):
        return self._eval(ngen)

    def evolve_population(self, pop, pop2, bound, f, c):
        npop, ndim = pop.shape

        for i in range(npop):

            # --- Vector selection ---
            v1, v2, v3 = i, i, i
            while v1 == i:
                v1 = randint(npop)
            while (v2 == i) or (v2 == v1):
                v2 = randint(npop)
            while (v3 == i) or (v3 == v2) or (v3 == v1):
                v3 = randint(npop)

            # --- Mutation ---
            v = pop[v1] + f * (pop[v2] - pop[v3])
            # random choice a value between when the solution out of the bounds
            for a, b in zip(enumerate(v), bound):
                if a[1] > b[1] or a[1] < b[0]:
                    v[a[0]] = np.random.uniform(b[0], b[1], 1)

            # --- Cross over ---
            co = rand(ndim)
            for j in range(ndim):
                if co[j] <= c:
                    pop2[i, j] = v[j]
                else:
                    pop2[i, j] = pop[i, j]

            # --- Forced crossing ---
            j = randint(ndim)
            pop2[i, j] = v[j]
        return pop2

    def _eval_sfun(self, ngen=1):
        """Run DE for a function that takes a single pv as an input and retuns a single value."""
        popc, fitc = self._population, self._fitness
        popt, fitt = self._trial_pop, self._trial_fit

        for ipop in range(self.n_pop):
            fitc[ipop] = self.m * self.minfun(popc[ipop, :])

        for igen in range(ngen):
            f = self.f or uniform(*self.fbounds)
            c = self.c or uniform(*self.cbounds)

            popt = self.evolve_population(popc, popt, self.bounds, f, c)
            fitt[:] = self.m * array(list(self.map(self.minfun, popt)))

            msk = fitt < fitc
            popc[msk, :] = popt[msk, :]
            fitc[msk] = fitt[msk]

            self._minidx = argmin(fitc)
            if fitc.ptp() < self.min_ptp:
                break

            yield popc[self._minidx, :], fitc[self._minidx]


class UtilityFunction_(UtilityFunction):
    def __init__(self, kind, kappa, xi):
        super(UtilityFunction_, self).__init__(kind, kappa, xi)

    def utility(self, x, gp, y_min):
        if self.kind == 'ucb':
            return self._ucb_(x, gp, self.kappa)
        if self.kind == 'ei':
            return self._ei_(x, gp, y_min, self.xi)
        if self.kind == 'poi':
            return self._poi_(x, gp, y_min, self.xi)

    @staticmethod
    def _ucb_(x, gp, kappa):
        mean, std = gp.predict(x, return_std=True)
        return mean - kappa * std

    @staticmethod
    def _ei_(x, gp, y_min, xi):
        mean, std = gp.predict(x, return_std=True)
        z = (y_min - mean - xi) / std
        return -(y_min - mean - xi) * norm.cdf(z) - std * norm.pdf(z)

    @staticmethod
    def _poi_(x, gp, y_min, xi):
        mean, std = gp.predict(x, return_std=True)
        z = (y_min - mean - xi) / std
        return -norm.cdf(z)


class BayesianOptimization_(BayesianOptimization):
    def __init__(self, f, pbounds, random_state=None, verbose=0):
        super(BayesianOptimization_, self).__init__(f, pbounds, random_state, verbose)

    def _prime_queue_LHS(self, init_points):
        """Make sure there's something in the queue at the very beginning."""
        if self._queue.empty and self._space.empty:
            init_points = max(init_points, 1)
        LHS_points = self.LHSample(init_points, self._space.bounds)
        for point in LHS_points:
            self._queue.add(point)

    def LHSample(self, N, bounds, D=None):
        if D == None:
            D = bounds.shape[0]
        result = np.empty([N, D])
        temp = np.empty([N])
        d = 1.0 / N
        for i in range(D):
            for j in range(N):
                temp[j] = np.random.uniform(
                    low=j * d, high=(j + 1) * d, size=1)[0]
            np.random.shuffle(temp)
            for j in range(N):
                result[j, i] = temp[j]
        lower_bounds = bounds[:, 0]
        upper_bounds = bounds[:, 1]
        if np.any(lower_bounds > upper_bounds):
            print('bounds error')
            return None
        np.add(np.multiply(result,
                           (upper_bounds - lower_bounds),
                           out=result),
               lower_bounds,
               out=result)
        return result

    def load_LHS(self, path):
        X, fit = [], []
        with open(path, 'r') as f:
            l = f.readlines()
        l.pop(0)
        p1 = re.compile(r'[{](.*?)[}]', re.S)
        for i in range(0, len(l)):
            l[i] = l[i].rstrip('\n')
            s = re.findall(p1, l[i])[0]
            d = eval('{' + s + '}')
            X.append(np.array(list(d.values())))
            f = float(l[i].replace('{' + s + '}', '').split(' ')[2])
            fit.append(f)
        return X, fit

    def acq_min_CMA(self, ac, gp, y_min, bounds, random_state):
        x_seeds = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                       size=(bounds.shape[0]))
        options = {'tolfunhist': -1e+4, 'tolfun': -1e+4, 'ftarget': -1e+4, 'bounds': bounds.T.tolist(), 'maxiter': 1000,
                   'verb_log': 0, 'verb_time': False, 'verbose': -9}
        res = cma.fmin(lambda x: ac(x.reshape(1, -1), gp=gp, y_min=y_min), x_seeds, 0.25, options=options,
                       restarts=0, incpopsize=0, restart_from_best=False, bipop=False)
        x_min = res[0]
        return np.clip(x_min, bounds[:, 0], bounds[:, 1])

    def acq_min_DE(self, ac, gp, y_min, bounds, random_state, ngen=100, npop=45, f=0.4, c=0.3):
        de = DiffEvol(lambda x: ac(x.reshape(1, -1), gp=gp, y_min=y_min)[0], bounds, npop, f=f, c=c,
                      seed=random_state)
        de.optimize(ngen)
        print(de.minimum_value, de.minimum_location, de.minimum_index)
        x_min = de.minimum_location
        return np.clip(x_min, bounds[:, 0], bounds[:, 1])

    def suggest_(self, utility_function, opt_function):
        if len(self._space) == 0:
            return self._space.array_to_params(self._space.random_sample())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(self._space.params, self._space.target)
        suggestion = opt_function(
            ac=utility_function.utility,
            gp=self._gp,
            y_min=self._space.target.min(),
            bounds=self._space.bounds,
            random_state=self._random_state.randint(100000)
        )
        return self._space.array_to_params(suggestion)

    def guess_fixedpoint(self, utility_function, X):
        gauss = utility_function.utility(X, self._gp, self._space.target.min())
        return gauss

    def minimize(self,
                 LHS_path=None,
                 init_points=5,
                 is_LHS=False,
                 n_iter=25,
                 acq='ucb',
                 opt=None,
                 kappa=2.576,
                 xi=0.0,
                 **gp_params):
        """Mazimize your function"""
        self._prime_subscriptions()
        self.dispatch(Events.OPTMIZATION_START)
        if LHS_path == None:
            if is_LHS:
                self._prime_queue_LHS(init_points)
            else:
                self._prime_queue(init_points)
        else:
            X, fit = self.load_LHS(LHS_path)
            for x, eva in zip(X, fit):
                self.register(x, eva)
        if opt == None:
            opt = self.acq_min_DE
        self.set_gp_params(**gp_params)
        util = UtilityFunction_(kind=acq, kappa=kappa, xi=xi)
        iteration = 0
        while not self._queue.empty or iteration < n_iter:
            try:
                x_probe = next(self._queue)
            except StopIteration:
                x_probe = self.suggest_(util, opt)
                iteration += 1
            self.probe(x_probe, lazy=False)
        self.dispatch(Events.OPTMIZATION_END)


class CoE():
    def __init__(self, f, f_p, SubCom, ranges, borders, precisions, acquisition, kappa=2.576, xi=0.0, **opts):
        self.aimfunc = f
        self.punfunc = f_p
        self.surrogate = BayesianOptimization_(
            f=f,
            pbounds=opts['bounds'], # 此处需要修改到和borders匹配的形式
            random_state=1,
        )
        self.util = UtilityFunction_(kind=acquisition, kappa=kappa, xi=xi)
        # opts['bounds'] = self.optimizer._space._bounds.T.tolist()

        self.SubCom = SubCom
        self.FieldDR = ga.crtfld(ranges, borders, precisions)

    def surrogate_init(self, init_points, LHS_path=None):
        if LHS_path == None:
            LHS_points = self.surrogate.LHSample(init_points.astype(int), self.surrogate._space.bounds)  # LHS for BO
            fit_init = [self.aimfunc(**self.surrogate._space.array_to_params(x)) for x in # 还是要和GA用的函数形式匹配一下
                        LHS_points]  # evaluated by the real fitness
            for x, eva in zip(LHS_points, fit_init):
                self.optimizer._space.register(x, eva)  # add LHS points to solution space
        else:
            LHS_points, fit_init = self.surrogate.load_LHS(LHS_path)
            for x, eva in zip(LHS_points, fit_init):
                self.optimizer._space.register(x, eva)  # add loaded LHS points to solution space
        self.surrogate._gp.fit(self.optimizer._space.params, self.optimizer._space.target)  # initialize the BO model

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
                [ObjVSel, LegVSel] = self.aimfuc(Chrom, LegVSel)  # 求育种种群的目标函数值
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