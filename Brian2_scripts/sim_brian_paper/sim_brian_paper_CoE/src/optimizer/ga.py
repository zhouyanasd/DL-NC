import numpy as np
from decimal import Decimal
import matplotlib.pyplot as plt

def rep(MatIn,REPN):
    N  = MatIn.shape
    # Calculate
    Ind_D = np.remainder(np.arange(0,REPN[0]*N[0]),N[0])
    Ind_L = np.remainder(np.arange(0,REPN[1]*N[1]),N[1])

    # Create output matrix
    MatOut = np.zeros((REPN[0]*N[0], REPN[1]*N[1]), dtype=MatIn.dtype)
    for i, ind_d in enumerate(Ind_D):
        for j, ind_l in enumerate(Ind_L):
            MatOut[i,j] = MatIn[ind_d,ind_l]
    return MatOut

def get_decimals_bit(value):
    value_p = str(value).split(".")
    try:
        if float(value_p[1]) == 0:
            return int(0)
        else:
            return len(value_p[1])
    except IndexError:
        return int(0)

def round_lb(value, precision, border):
    value_p = get_decimals_bit(value)
    value_str = str(value+0.1**precision*abs(bool(border)-1)) if precision >= value_p else str(value)
    precision_str = '0.'+''.join(['0']*precision)
    rounding = "ROUND_CEILING" if abs(bool(border)-1) else "ROUND_FLOOR"
    value_deal = Decimal(value_str).quantize(Decimal(precision_str), rounding = rounding)
    return int(value_deal) if precision == 0 else float(value_deal)

def round_ub(value, precision, border):
    value_p = get_decimals_bit(value)
    value_str = str(value-0.1**precision*abs(bool(border)-1)) if precision >= value_p else str(value)
    precision_str = '0.'+''.join(['0']*precision)
    rounding = "ROUND_FLOOR" if abs(bool(border)-1) else "ROUND_CEILING"
    value_deal = Decimal(value_str).quantize(Decimal(precision_str), rounding = rounding)
    return int(value_deal) if precision == 0 else float(value_deal)

def crtfld(ranges, borders=None, precisions=None, codes=None, scales=None):
    dim = ranges.shape[1]
    lb = [round_lb(ranges[0][x],precisions[x], borders[0][x]) for x in range(dim)]
    ub = [round_ub(ranges[1][x],precisions[x], borders[1][x]) for x in range(dim)]
    if codes == None:
        FieldDR = np.array([lb, ub])
        return FieldDR
    else:
        scales = [0] * dim if scales is None else scales
        lbin = [1 if precisions[x] == 0 and borders[0][x]==0 else borders[0][x] for x in range(dim)]
        ubin = [1 if precisions[x] == 0 and borders[1][x]==0 else borders[1][x] for x in range(dim)]
        len_ = [int((x-y)*10**p).bit_length() for p,x,y in zip(precisions,ub,lb)]
        FieldD = np.array([len_, lb, ub, codes, scales, lbin, ubin], dtype=object)
        return FieldD

def crtrp(Nind,FieldDR):
    Nvar = FieldDR.shape[1]
    # Compute Matrix with Range of variables and Matrix with Lower value
    Range = rep((FieldDR[1,:]-FieldDR[0,:]).reshape(1,-1),[Nind,1])
    Lower =  rep(FieldDR[0,:].reshape(1,-1), [Nind,1])

    # Create initial population
    # Each row contains one individual, the values of each variable uniformly
    # distributed between lower and upper bound (given by FieldDR)
    Chrom = np.random.rand(Nind,Nvar) * Range + Lower
    return Chrom

def crtip(Nind,FieldDR):
    Chrom = crtrp(Nind,FieldDR)
#     func = lambda x : int(Decimal(str(x)).quantize(Decimal('0.'), rounding = 'ROUND_HALF_EVEN'))
#     return np.frompyfunc(func,1,1)(Chrom).astype(np.int64)
    return np.floor(Chrom).astype(np.int64)

def crtbase(Lind, Base=2):
    if type(Lind) == int and type(Base) == int:
        LenL = 1
        Lind_ = np.array([Lind])
        Base_ =  Base * np.ones(LenL, dtype=np.int)
    elif type(Lind) == np.ndarray and type(Base) == int:
        ml, LenL = Lind.shape
        Lind_ = Lind[0]
        Base_ = Base * np.ones(LenL, dtype=np.int)
    elif type(Lind) == np.ndarray and type(Base) == np.ndarray:
        ml, LenL = Lind.shape
        mb, LenB = Base.shape
        if LenL == LenB:
            Base_ = Base[0]
            Lind_ = Lind[0]
        else:
            raise ValueError('Vector dimensions must agree')
    elif type(Lind) == int and type(Base) == np.ndarray:
        LenL = int(Lind)
        mb, LenB = Base.shape
        if LenL == LenB:
            Base_ = Base[0]
            Lind_ = np.ones(LenL, dtype=np.int)
        else:
            raise ValueError('Vector dimensions must agree')
    else:
        raise ValueError('Lind or Base must be numpy.array in shape (1,2) or int')
    BaseV = []
    for i in range(LenL):
        BaseV.extend([Base_[i]]*Lind_[i])
    return np.array(BaseV, dtype=np.int64).reshape(1,-1)


def crtbp(Nind, LorB):
    if type(LorB) is int:
        BaseV = crtbase(LorB)
        Lind = LorB
    elif type(LorB) is np.ndarray:
        BaseV = LorB
        Lind = LorB.shape[1]

    Chrom = np.random.rand(Nind, Lind) * BaseV[np.zeros(Nind, dtype=np.int), :]
    return np.floor(Chrom).astype(np.int64)


def reclin(OldChrom, XOVR=None):
    NewChrom = np.zeros(OldChrom.shape)
    Nind, Nvar = OldChrom.shape
    Xops = int(np.floor(Nind / 2))

    odd = np.arange(1, Nind, 2)
    even = np.arange(0, Nind - 1, 2)

    # position of value of offspring compared to parents
    Alpha = -0.25 + 1.5 * np.random.rand(Xops, 1)
    Alpha = Alpha[0:Xops, np.zeros(Nvar, dtype=np.int)]

    # recombination
    NewChrom[odd, :] = OldChrom[odd, :] + Alpha * (OldChrom[even, :] - OldChrom[odd, :])

    # the same ones more for second half of offspring
    Alpha = -0.25 + 1.5 * np.random.rand(Xops, 1)
    Alpha = Alpha[0:Xops, np.zeros(Nvar, dtype=np.int)]
    NewChrom[even, :] = OldChrom[odd, :] + Alpha * (OldChrom[even, :] - OldChrom[odd, :])

    if np.remainder(Nind, 2):
        NewChrom[Nind - 1, :] = OldChrom[Nind - 1, :]

    return NewChrom


def recint(OldChrom, XOVR=None):
    NewChrom = np.zeros(OldChrom.shape)
    Nind, Nvar = OldChrom.shape
    Xops = int(np.floor(Nind / 2))

    odd = np.arange(1, Nind, 2)
    even = np.arange(0, Nind - 1, 2)

    # position of value of offspring compared to parents
    Alpha = -0.25 + 1.5 * np.random.rand(Xops, Nvar)

    # recombination
    NewChrom[odd, :] = OldChrom[odd, :] + Alpha * (OldChrom[even, :] - OldChrom[odd, :])

    # the same ones more for second half of offspring
    Alpha = -0.25 + 1.5 * np.random.rand(Xops, Nvar)
    NewChrom[even, :] = OldChrom[odd, :] + Alpha * (OldChrom[even, :] - OldChrom[odd, :])

    if np.remainder(Nind, 2):
        NewChrom[Nind - 1, :] = OldChrom[Nind - 1, :]

    return NewChrom


def recdis(OldChrom, XOVR=None):
    NewChrom = np.zeros(OldChrom.shape, dtype=OldChrom.dtype)
    Nind, Nvar = OldChrom.shape
    Xops = int(np.floor(Nind / 2))

    # which parent gives the value
    Mask1 = (np.random.rand(Xops, Nvar) < 0.5)
    Mask2 = (np.random.rand(Xops, Nvar) < 0.5)

    odd = np.arange(1, Nind, 2)
    even = np.arange(0, Nind - 1, 2)

    NewChrom[odd, :] = (OldChrom[odd, :] * Mask1) + (OldChrom[even, :] * (1 - Mask1))
    NewChrom[even, :] = (OldChrom[odd, :] * Mask2) + (OldChrom[even, :] * (1 - Mask2))
    return NewChrom


def xovmp(OldChrom, Px=0.7, Npt=0, Rs=0):
    Nind, Lind = OldChrom.shape
    if Lind < 2:
        return OldChrom
    Xops = int(Decimal(str(Nind / 2)).quantize(Decimal('0.'), rounding='ROUND_FLOOR'))
    DoCross = np.random.rand(Xops) < Px
    odd = np.arange(1, Nind, 2)
    even = np.arange(0, Nind - 1, 2)
    Mask = (1 - bool(Rs)) | (OldChrom[odd, :] != OldChrom[even, :])
    Mask = np.cumsum(Mask, axis=1)
    # 根据有效长度和Px计算每对个体的交叉位点(两个相等的交叉位点表示没有交叉)
    xsites = np.zeros((Mask.shape[0], 2), dtype=np.int32)
    xsites[:, 0] = Mask[:, Lind - 1]
    if Npt >= 2:
        xsites[:, 0] = np.ceil(xsites[:, 0] * np.random.rand(Xops))
    xsites[:, 1] = np.remainder((xsites[:, 0] +
                                 np.ceil((Mask[:, Lind - 1] - 1) * np.random.rand(Xops)) * DoCross - 1),
                                Mask[:, Lind - 1]) + 1

    # Express cross sites in terms of a 0-1 mask
    Mask_finall = (xsites[:, np.zeros(Lind, dtype=np.int)] < Mask) == (xsites[:, np.ones(Lind, dtype=np.int)] < Mask)
    if 1 - bool(Npt):
        shuff = np.random.rand(Lind, Xops)
        order = np.argsort(shuff, axis=0)
        for i in range(Xops):
            OldChrom[odd[i], :] = OldChrom[odd[i], order[:, i]]
            OldChrom[even[i], :] = OldChrom[even[i], order[:, i]]
    # Perform crossover
    NewChrom = np.zeros(OldChrom.shape, dtype=np.int64)
    NewChrom[odd, :] = OldChrom[odd, :] * Mask_finall + OldChrom[even, :] * (1 - Mask_finall)
    NewChrom[even, :] = OldChrom[odd, :] * (1 - Mask_finall) + OldChrom[even, :] * Mask_finall
    # If the number of individuals is odd, the last individual cannot be mated
    # but must be included in the new population
    if np.remainder(Nind, 2):
        NewChrom[Nind - 1, :] = OldChrom[Nind - 1, :]

    if 1 - bool(Npt):
        re_order = np.argsort(order, axis=0)
        for i in range(Xops):
            NewChrom[odd[i], :] = NewChrom[odd[i], re_order[:, i]]
            NewChrom[even[i], :] = NewChrom[even[i], re_order[:, i]]
    return NewChrom

def xovdp(OldChrom, XOVR=None):
    return xovmp(OldChrom, XOVR)


def recombin(REC_F, Chrom, RecOpt=0.7, SUBPOP=1):
    Nind, Nvar = Chrom.shape
    # Select individuals of one subpopulation and call low level function
    NewChrom = np.array([]).reshape(-1, Nvar)
    for irun in range(SUBPOP):
        ChromSub = Chrom[irun * Nind:(irun + 1) * Nind, :]
        NewChromSub = globals()[REC_F](ChromSub, RecOpt)
        NewChrom = np.append(NewChrom, NewChromSub, axis=0)
    return NewChrom


def mutbga(OldChrom, FieldDR, Pm=None, MutShrink=1, Gradient=20):
    Nind, Nvar = OldChrom.shape
    mF, nF = FieldDR.shape
    if Pm == None:
        Pm = 0.7 / Nind
    # Matrix with range values for every variable
    Range = rep(0.5 * MutShrink * (FieldDR[1, :] - FieldDR[0, :]).reshape(1, -1), [Nind, 1])
    # zeros and ones for mutate or not this variable, together with Range
    Range = Range * (np.random.rand(Nind, Nvar) < Pm)
    # compute, if + or - sign
    Range = Range * (1 - 2 * (np.random.rand(Nind, Nvar) < 0.5))
    # used for later computing, here only ones computed
    ACCUR = Gradient
    Vect = np.array([2 ** (-x) for x in range(ACCUR)])
    Delta = (np.random.rand(Nind, ACCUR) < 1 / ACCUR).dot(Vect)
    Delta = rep(Delta.reshape(-1, 1), [1, Nvar])
    # perform mutation
    NewChrom = OldChrom + Range * Delta
    # Ensure variables boundaries, compare with lower and upper boundaries
    np.clip(NewChrom, rep(FieldDR[0, :].reshape(1, -1), [Nind, 1]), rep(FieldDR[1, :].reshape(1, -1), [Nind, 1]),
            NewChrom)
    return NewChrom


def mutint(OldChrom, FieldDR, Pm=None, params3=None, params4=None):
    Nind, Nvar = OldChrom.shape
    mF, nF = FieldDR.shape
    if Pm == None:
        Pm = 0.7 / Nind
    # Matrix with range values for every variable
    Range = rep(0.5 * (FieldDR[1, :] - FieldDR[0, :]).reshape(1, -1), [Nind, 1])
    # zeros and ones for mutate or not this variable, together with Range
    Range = Range * (np.random.rand(Nind, Nvar) < Pm)
    # compute, if + or - sign
    Range = Range * (1 - 2 * (np.random.rand(Nind, Nvar) < 0.5))
    # perform mutation
    NewChrom = OldChrom + Range
    # Ensure variables boundaries, compare with lower and upper boundaries
    np.clip(np.round(NewChrom), rep(FieldDR[0, :].reshape(1, -1), [Nind, 1]),
            rep(FieldDR[1, :].reshape(1, -1), [Nind, 1]), NewChrom)
    return NewChrom


def mutbin(OldChrom, Pm=None, params3=None, params4=None):
    Nind, Nvar = OldChrom.shape
    FieldDR = np.array([[0] * Nvar, [1] * Nvar])

    NewChrom = mutint(OldChrom, FieldDR, Pm)

    return NewChrom.astype(np.int64)


def ranking(ObjV, LegV, RFun=None, SUBPOP=1):
    if ObjV.shape != LegV.shape:
        raise ValueError('The ObjV and LegV should be match.')

    for i, o in enumerate(ObjV):
        if o == None or o == np.nan:
            LegV[i] = 0

    Nind, ans = ObjV.shape

    if RFun is None:
        RFun = np.array([[2]], dtype=np.int)

    if RFun.shape == (1, 1):
        RFun_ = RFun[0][0]
        NonLin = 0
    elif RFun.shape == (1, 2):
        RFun_ = RFun[0][0]
        NonLin = RFun[0][1]
    elif RFun.shape[0] == Nind:
        RFun_ = RFun
    else:
        raise ValueError('RFun disagree')

    if type(SUBPOP) != int:
        raise ValueError('SUBPOP must be a scalar')

    if Nind % SUBPOP == 0:
        Nind_ = int(Nind / SUBPOP)
    else:
        raise ValueError('ObjV and SUBPOP disagree')

    if RFun_ is not np.ndarray:
        if NonLin == 0:
            # linear ranking with SP between 1 and 2
            if RFun_ < 1 or RFun_ > 2:
                raise ValueError('Selective pressure for linear ranking must be between 1 and 2');
            else:
                RFun_ = 2 - RFun_ + 2 * (RFun_ - 1) * np.arange(0, Nind_) / (Nind_ - 1)
        elif NonLin == 1:
            if RFun_ < 1:
                raise ValueError('Selective pressure must be greater than 1')
            elif RFun_ > Nind - 2:
                raise ValueError('Selective pressure too big')
            else:

                Root1 = np.roots(np.array([RFun_ - Nind_] + [RFun_] * (Nind_ - 1)))
                RFun_ = np.power(abs(Root1[0]) * np.ones(Nind_), np.arange(0, Nind_))
                RFun_ = RFun_ / sum(RFun_) * Nind_

    FitnV = []
    # loop over all subpopulations
    for irun in range(SUBPOP):
        # Copy objective values of actual subpopulation
        ObjVSub = ObjV[irun * Nind_:(irun + 1) * Nind_].reshape(-1)
        # Sort does not handle NaN values as required. So, find those...
        NaNix = np.isnan(ObjVSub.astype(float))
        Validix = 1 - NaNix
        # ... and sort only numeric values (smaller is better).
        ix = np.argsort(np.argsort(-ObjVSub[np.where(Validix == 1)[0]]))

        # Now build indexing vector assuming NaN are worse than numbers,
        # (including Inf!)...
        ix = np.append(np.where(Validix == 0)[0], ix)
        # Add FitnVSub to FitnV
        FitnV = np.append(FitnV, RFun_[ix])
    return FitnV.reshape(-1, 1)

def sus(FitnV, Nsel):
    Nind,ans = FitnV.shape
    cumfit = np.cumsum(FitnV)
    trials = cumfit[Nind-1] / Nsel* np.random.rand()+np.arange(Nsel)
    Mf = rep(cumfit.reshape(-1,1),[1, Nsel])
    Mt = rep(trials.reshape(1,-1),[Nind, 1])
    ChIndex = np.sum((Mt < Mf ) & (np.append(np.zeros((1, Nsel)), Mf[0:Nind-1, :], axis=0)<= Mt), axis=1)
    NewChrIx = []
    for i, c in enumerate(ChIndex):
        while c > 0:
            NewChrIx.append(i)
            c -= 1
    NewChrIx = np.array(NewChrIx, dtype=np.int64)
    np.random.shuffle(NewChrIx)
    return NewChrIx

def rws(FitnV, Nsel):
    Nind,ans = FitnV.shape
    cumfit = np.cumsum(FitnV)
    trials = cumfit[Nind-1] * np.random.rand(Nsel)
    Mf = rep(cumfit.reshape(-1,1),[1, Nsel])
    Mt = rep(trials.reshape(1,-1),[Nind, 1])
    ChIndex = np.sum((Mt < Mf ) & (np.append(np.zeros((1, Nsel)), Mf[0:Nind-1, :], axis=0)<= Mt), axis=1)
    NewChrIx = []
    for i, c in enumerate(ChIndex):
        while c > 0:
            NewChrIx.append(i)
            c -= 1
    NewChrIx = np.array(NewChrIx, dtype=np.int64)
    return NewChrIx


def tour(FitnV, Nsel):
    Nind, ans = FitnV.shape
    tour = int(np.ceil(FitnV.max()))
    if tour > Nind:
        tour = int(np.ceil(FitnV.mean()))
    if tour > Nind:
        tour = 2
    NewChrIx = []
    for i in range(Nsel):
        FitnV_ = np.random.choice(FitnV.reshape(-1), tour, replace=True)
        ChrIx = np.where(FitnV.reshape(-1) == FitnV_.max())[0][0]
        NewChrIx.append(ChrIx)
    return np.array(NewChrIx, dtype=np.int)


def etour(FitnV, Nsel):
    Nind, ans = FitnV.shape
    tour = int(np.ceil(FitnV.max()))
    if tour > Nind:
        tour = int(np.ceil(FitnV.mean()))
    if tour > Nind:
        tour = 2
    NewChrIx = [np.where(FitnV.reshape(-1) == FitnV.max())[0][0]]
    for i in range(Nsel - 1):
        FitnV_ = np.random.choice(FitnV.reshape(-1), tour, replace=True)
        ChrIx = np.where(FitnV.reshape(-1) == FitnV_.max())[0][0]
        NewChrIx.append(ChrIx)
    NewChrIx = np.array(NewChrIx, dtype=np.int)
    np.random.shuffle(NewChrIx)
    return NewChrIx


def selecting(SEL_F, Chrom, FitnV, GGAP=1.0, SUBPOP=1, ObjV=None, LegV=None):
    # Identify the population size (Nind)
    NindCh, Nvar = Chrom.shape
    NindF, VarF = FitnV.shape
    if NindCh != NindF:
        raise ValueError('Chrom and FitnV disagree')
    if VarF != 1:
        raise ValueError('FitnV must be a column vector')
    if NindCh % SUBPOP == 0:
        Nind_ = int(NindCh / SUBPOP)
    else:
        raise ValueError('ObjV and SUBPOP disagree')
    # Compute number of new individuals (to select)
    NSel = int(max(np.floor(Nind_ * GGAP + .5), 2))
    # Select individuals from population
    SelCh = np.array([]).reshape(-1, Nvar)
    ChrIx = np.array([], dtype=np.int)
    for irun in range(SUBPOP):
        FitnVSub = FitnV[irun * Nind_:(irun + 1) * Nind_]
        ChrIx_ = globals()[SEL_F](FitnVSub, NSel) + irun * Nind_
        SelCh = np.append(SelCh, Chrom[ChrIx_, :], axis=0)
        ChrIx = np.append(ChrIx, ChrIx_)
    if ObjV is None and LegV is None:
        return SelCh
    elif ObjV is not None and LegV is None:
        return SelCh, ObjV[ChrIx]
    elif ObjV is not None and LegV is not None:
        return SelCh, ObjV[ChrIx], LegV[ChrIx]


def bs2rv(Chrom, FieldD):
    Chrom = Chrom.copy()
    FieldD = FieldD.copy()
    Nind, Lind = Chrom.shape
    seven, Nvar = FieldD.shape
    if seven != 7:
        raise ValueError('FieldD must have 7 rows.')
    # Get substring properties
    length = FieldD[0, :]
    lb = FieldD[1, :].astype(float)
    ub = FieldD[2, :].astype(float)
    code = FieldD[3, :].astype(bool)
    scale = FieldD[4, :].astype(bool)
    lin = FieldD[5, :].astype(float)
    uin = FieldD[6, :].astype(float)
    # Check substring properties for consistency
    if sum(length) != Lind:
        raise ValueError('Data in FieldD must agree with chromosome length')
    if 1 - (lb[scale] * ub[scale] > 0).all():
        raise ValueError('Log-scaled variables must not include 0 in their range')
    # Decode chromosomes
    Phen = np.zeros((Nind, Nvar))
    lf = np.cumsum(length) - 1
    li = np.cumsum(np.array([0] + list(length)))
    Prec = 0.5 ** length
    logsgn = np.sign(lb[scale])
    lb[scale] = np.log(abs(lb[scale]))
    ub[scale] = np.log(abs(ub[scale]))
    delta = ub - lb
    num = (1 - lin) * Prec
    den = (lin + uin - 1) * Prec
    for i in range(Nvar):
        idx = np.arange(li[i], lf[i] + 1)
        if code[i]:  # Gray decoding
            Chrom[:, idx] = np.remainder(np.cumsum(Chrom[:, idx].T, axis=0).T, 2)
        Phen[:, i:i + 1] = np.dot(Chrom[:, idx], ((.5) ** np.arange(1, length[i] + 1).reshape(-1, 1)))
        Phen[:, i] = lb[i] + np.dot(delta[i], (Phen[:, i] + num[i]) / (1 - den[i]))
    expand = np.zeros(Nind, dtype=np.int)
    if scale.any():
        Phen[:, scale] = logsgn[expand].reshape(-1, 1) * np.exp(Phen[:, scale])
    return Phen


def bs2int(Chrom, FieldD):
    FieldD_ = FieldD.copy()
    FieldD_[4, :] = np.array([0, 0])
    FieldD_[5, :] = np.array([1, 1])
    FieldD_[6, :] = np.array([1, 1])
    Phen = bs2rv(Chrom, FieldD_)
    return np.fix(Phen).astype(np.int64)

def is2(FieldD):
    r = FieldD[2, :] - FieldD[1, :]
    result = [dec2bin(x, l) for x, l in zip(r, FieldD[0, :])]
    result = [i for item in result for i in item]
    if (np.array(result) == 1).all():
        return True
    else:
        return False

def rv2bs(gen, FieldD):
    result = []
    for individual in gen:
        gen_i = []
        for g, u, c, l in zip(individual, FieldD[1,:], FieldD[3, :], FieldD[0, :]):
            g_b = dec2bin(g-u, l)
            if c == 1:
                g_g = bin2gary(g_b)
                gen_i.extend(g_g)
            elif c == 0:
                gen_i.extend(g_b)
        result.append(gen_i)
    return np.array(result)

def bin2dec(binary):
    result = 0
    for i in range(len(binary)):
        result += int(binary[-(i + 1)]) * pow(2, i)
    return result

def gray2bin(gray):
    result = []
    result.append(gray[0])
    for i, g in enumerate(gray[1:]):
        result.append(g ^ result[i])
    return result

def dec2bin(num, l):
    result = []
    while True:
        num, remainder = divmod(num, 2)
        result.append(int(remainder))
        if num == 0:
            break
    if len(result) < l:
        result.extend([0] * (l - len(result)))
    return result[::-1]

def bin2gary(binary):
    result = []
    result.append(binary[0])
    for i, b in enumerate(binary[1:]):
        result.append(b ^ binary[i])
    return result

def trcplot(pop_trace, labels, titles = None, save_path = None):
    l = len(pop_trace)
    t = np.arange(l)
    index = 0
    for i, l_i in enumerate(labels):
        plt.figure()
        plt.xlabel('Iteration')
        plt.grid(True)
        for l_j in l_i:
            plt.plot(t, pop_trace[:,index:index+1], label=l_j)
            index += 1
        plt.legend()
        if titles is not None:
            plt.title(titles[i])
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()