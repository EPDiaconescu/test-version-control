

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import copy, sort, amax, arange, exp, sqrt, abs, floor, searchsorted
from scipy.misc import factorial, comb
import itertools
import statsmodels.api as sm



def kuiper_FPP(D,N):
    """Compute the false positive probability for the Kuiper statistic.

    Uses the set of four formulas described in Paltani 2004; they report
    the resulting function never underestimates the false positive probability
    but can be a bit high in the N=40..50 range. (They quote a factor 1.5 at
    the 1e-7 level.

    Parameters
    ----------
    D : float
        The Kuiper test score.
    N : float
        The effective sample size.

    Returns
    -------
    fpp : float
        The probability of a score this large arising from the null hypothesis.

    Reference
    ---------
    Paltani, S., "Searching for periods in X-ray observations using
    Kuiper's test. Application to the ROSAT PSPC archive", Astronomy and
    Astrophysics, v.240, p.789-790, 2004.

    """
    if D<0. or D>2.:
        raise ValueError("Must have 0<=D<=2 by definition of the Kuiper test")

    if D<2./N:
        return 1. - factorial(N)*(D-1./N)**(N-1)
    elif D<3./N:
        k = -(N*D-1.)/2.
        r = sqrt(k**2 - (N*D-2.)/2.)
        a, b = -k+r, -k-r
        return 1. - factorial(N-1)*(b**(N-1.)*(1.-a)-a**(N-1.)*(1.-b))/float(N)**(N-2)*(b-a)
    elif (D>0.5 and N%2==0) or (D>(N-1.)/(2.*N) and N%2==1):
        def T(t):
            y = D+t/float(N)
            return y**(t-3)*(y**3*N-y**2*t*(3.-2./N)/N-t*(t-1)*(t-2)/float(N)**2)
        s = 0.
        # NOTE: the upper limit of this sum is taken from Stephens 1965
        for t in xrange(int(floor(N*(1-D)))+1):
            term = T(t)*comb(N,t)*(1-D-t/float(N))**(N-t-1)
            s += term
        return s
    else:
        z = D*sqrt(N)
        S1 = 0.
        term_eps = 1e-12
        abs_eps = 1e-100
        for m in itertools.count(1):
            T1 = 2.*(4.*m**2*z**2-1.)*exp(-2.*m**2*z**2)
            so = S1
            S1 += T1
            if abs(S1-so)/(abs(S1)+abs(so))<term_eps or abs(S1-so)<abs_eps:
                break
        S2 = 0.
        for m in itertools.count(1):
            T2 = m**2*(4.*m**2*z**2-3.)*exp(-2*m**2*z**2)
            so = S2
            S2 += T2
            if abs(S2-so)/(abs(S2)+abs(so))<term_eps or abs(S1-so)<abs_eps:
                break
        return S1 - 8*D/(3.*sqrt(N))*S2

def kuiper(data, cdf=lambda x: x, args=()):
    """Compute the Kuiper statistic.

    Use the Kuiper statistic version of the Kolmogorov-Smirnov test to
    find the probability that something like data was drawn from the
    distribution whose CDF is given as cdf.

    Parameters
    ----------
    data : array-like
        The data values.
    cdf : callable
        A callable to evaluate the CDF of the distribution being tested
        against. Will be called with a vector of all values at once.
    args : list-like, optional
        Additional arguments to be supplied to cdf.

    Returns
    -------
    D : float
        The raw statistic.
    fpp : float
        The probability of a D this large arising with a sample drawn from
        the distribution whose CDF is cdf.

    Notes
    -----
    The Kuiper statistic resembles the Kolmogorov-Smirnov test in that
    it is nonparametric and invariant under reparameterizations of the data.
    The Kuiper statistic, in addition, is equally sensitive throughout
    the domain, and it is also invariant under cyclic permutations (making
    it particularly appropriate for analyzing circular data).

    Returns (D, fpp), where D is the Kuiper D number and fpp is the
    probability that a value as large as D would occur if data was
    drawn from cdf.

    Warning: The fpp is calculated only approximately, and it can be
    as much as 1.5 times the true value.

    Stephens 1970 claims this is more effective than the KS at detecting
    changes in the variance of a distribution; the KS is (he claims) more
    sensitive at detecting changes in the mean.

    If cdf was obtained from data by fitting, then fpp is not correct and
    it will be necessary to do Monte Carlo simulations to interpret D.
    D should normally be independent of the shape of CDF.

    """

    # FIXME: doesn't work for distributions that are actually discrete (for example Poisson).
    data = sort(data)
    cdfv = cdf(data,*args)
    N = len(data)
    D = amax(cdfv-arange(N)/float(N)) + amax((arange(N)+1)/float(N)-cdfv)

    return D, kuiper_FPP(D,N)

def kuiper_two(data1, data2):
    """Compute the Kuiper statistic to compare two samples.

    Parameters
    ----------
    data1 : array-like
        The first set of data values.
    data2 : array-like
        The second set of data values.

    Returns
    -------
    D : float
        The raw test statistic.
    fpp : float
        The probability of obtaining two samples this different from
        the same distribution.

    Notes
    -----
    Warning: the fpp is quite approximate, especially for small samples.

    """
    data1, data2 = sort(data1), sort(data2)

    if len(data2)<len(data1):
        data1, data2 = data2, data1

    cdfv1 = searchsorted(data2, data1)/float(len(data2)) # this could be more efficient
    cdfv2 = searchsorted(data1, data2)/float(len(data1)) # this could be more efficient
    D = (amax(cdfv1-arange(len(data1))/float(len(data1))) +
            amax(cdfv2-arange(len(data2))/float(len(data2))))

    Ne = len(data1)*len(data2)/float(len(data1)+len(data2))
    return D, kuiper_FPP(D, Ne)


#################################################
domeniu='ARCcordex'

#list_varName=['FD','GDD5','HDD17','prAnualMean','pr1mm','PrCum_GTp95','PrCum_GTp99','PrTotPc_GTp95','PrTotPc_GTp99',
#             'rx1day','rx5days','SU15','tasAnualMean','TN10_K','TN10p','TNn_K','TX90_K','TX90p','TXx_K']
list_varName=['rx1day']
periode='1980to2004'
ValnrYY='15'
fyy='1980'
lyy='2004'
output='D:/NowWorking/newProjet/results/domARC_CORDEX_b/KuiperPerkins_example/'
input='D:/NowWorking/newProjet/results/domARC_CORDEX_b/simSelected_ref/'

for varName in list_varName:
    inputObs='D:/NowWorking/newProjet/results/domARC_CORDEX_b/stations_Anom_ref/'+varName+'_StationsAnomalies_ARCcordex_15noYY_1980to2004_.csv'
    inputObsM='D:/NowWorking/newProjet/results/domARC_CORDEX_b/stations_climMean_ref/'+varName+'_StationsAnomalies_ARCcordex_15noYY_1980to2004_.csv'

    if varName=='FD':
        increment=2
    if varName=='GDD5':
        increment=20
    if varName=='HDD17':
        increment=300
    if varName=='prAnualMean':
        increment=0.2
    if varName=='pr1mm':
        increment=5
    if varName=='PrCum_GTp95':
        increment=20
    if varName=='PrTotPc_GTp95':
        increment=2
    if varName=='PrCum_GTp99':
        increment=20
    if varName=='PrTotPc_GTp99':
        increment=2
    if varName=='rx1day':
        increment=2
    if varName=='rx5days':
        increment=3
    if varName=='SU15':
        increment=1
    if varName=='tasAnualMean':
        increment=0.5
    if varName=='TN10_K':
        increment=0.5
    if varName=='TN10p':
        increment=1
    if varName=='TNn_K':
        increment=1.0
    if varName=='TX90_K':
        increment=0.5
    if varName=='TX90p':
        increment=1
    if varName=='TXx_K':
        increment=0.5

    ###################################################################
    df_Mean=pd.read_csv(inputObsM, sep=',')
    obs_Mean=df_Mean.iloc[:,1:]
    obs_Mean.index=['Obs','lat','lon']
    df_coord=pd.read_csv(inputObs, sep=',')
    ec_id=df_coord.columns[1:]

    data_obs=df_coord.iloc[:,1:]
    yy = df_coord.iloc[:,0]
    data_obs.index=yy

    listMM=['AWI-HIRHAM5_ARC44_MPI-ESM-LR_histo_']
    listMM2=['Obs', 'AWI-HIRHAM5-MPI-ESM-LR']


    allDataPD=data_obs[ec_id[0]]
    for file in listMM:
        df_model=pd.read_csv(input+varName+'_'+file+domeniu+'_'+str(ValnrYY)+'noYY_'+periode+'_spTempo.csv', sep=',')
        yyMM = allDataPD.index
        df_model.index=yyMM
        data_sh_1D=df_model[ec_id[0]]
        allDataPD = pd.concat([allDataPD, data_sh_1D], axis=1)

    allDataPD.columns=listMM2
    newFrame=allDataPD.dropna(axis=0)
    bbb=newFrame.convert_objects(convert_numeric=True)
    aaa=bbb.describe()
    ccc=(aaa.transpose())['mean']
    newFrame_dt=bbb-ccc
    newFrame_mean=pd.DataFrame({ec_id[0]:ccc})

    for i in range(1,len(ec_id)):
        allDataPDb=data_obs[ec_id[i]]

        for file in listMM:
            df_model=pd.read_csv(input+varName+'_'+file+domeniu+'_'+str(ValnrYY)+'noYY_'+periode+'_spTempo.csv', sep=',')
            yyMM = allDataPDb.index
            df_model.index=yyMM
            data_sh_1D=df_model[ec_id[i]]
            allDataPDb = pd.concat([allDataPDb, data_sh_1D], axis=1)

        allDataPDb.columns=listMM2
        newFrameb=allDataPDb.dropna(axis=0)
        bbb=newFrameb.convert_objects(convert_numeric=True)
        aaa=bbb.describe()
        ccc=(aaa.transpose())['mean']
        newFrameb_dt=bbb-ccc
        ddd=pd.DataFrame({ec_id[i]:ccc})
        newFrame_mean=pd.concat([newFrame_mean, ddd], axis=1)

        newFrame_dt=np.vstack([newFrame_dt, newFrameb_dt])


        newFrame_F=pd.DataFrame(newFrame_dt)
        newFrame_F.columns=newFrame.columns

    newFrame_mean2=newFrame_mean.drop('Obs', axis=0)
    meanFinal=pd.concat([newFrame_mean2.transpose(),obs_Mean.transpose()], axis=1)

    (tt, model)=newFrame_F.shape
    Dval = np.zeros(model)
    Pkval=np.ones(model)
    x=np.array(newFrame_F.iloc[:,0],dtype=float)
    ecdfX = sm.distributions.ECDF(x)

    for mm in range(1,model):
        y=np.array(newFrame_F.iloc[:,mm])
        ecdfY = sm.distributions.ECDF(y)
        D, p = kuiper_two(x, y)
        Dval[mm]=D
#        sns.set_context("poster")
#        sns.set_style("whitegrid")
        fig1 = plt.figure(figsize=(12,8))
        ax = fig1.add_subplot(111)
#        ax.patch.set_facecolor('white')
#        plt.grid(True,linestyle="-", color=[0.7,0.7,0.7])
        ax.plot( [10], [0.5], '.', color='white', label='D ='+str(D)[:4])
        ax.plot( ecdfX.x, ecdfX.y, linewidth = 3, color='blue',label='ECDF of observations')
        ax.plot( ecdfY.x, ecdfY.y, linewidth = 3, color='red', label='ECDF of '+newFrame.columns[mm])

        plt.xticks([-40,-20,0,20,40,60,80,100],fontsize=24)
        plt.xlabel("RX1day (mm/day)", fontsize=24)
        plt.ylabel("ECDF", fontsize=24)
        plt.yticks(fontsize=24)
        plt.xticks(fontsize=24)
        plt.title('a)',fontsize=24)

        lg=plt.legend(loc=4, fontsize=18)
        lg.draw_frame(False)

        plt.savefig(output+'Fig9a.png')

        fig1.clf()

        valmin=round(min(x.min(),y.min()))
        valmax=round(max(x.max(),y.max()))
        binsVal=int((valmax-valmin)/increment)
        valmax=valmin+increment*binsVal
        dist_space = np.linspace( valmin, valmax,binsVal+1)

        histX, binsX = np.histogram(x,bins=binsVal,range=(dist_space[0],dist_space[-1]) ,density=True)
        widthX = 0.95 * (binsX[1] - binsX[0])
        centerX = (binsX[:-1] + binsX[1:]) / 2

        histY, binsY = np.histogram(y,bins=binsVal,range=(dist_space[0],dist_space[-1]) ,density=True)
        widthY = 0.95 * (binsY[1] - binsY[0])
        centerY = (binsY[:-1] + binsY[1:]) / 2

        testM2=np.min(np.vstack((histX,histY)), axis=0)
        areaMin2 = np.sum(testM2)/np.sum(histX)
        Pkval[mm]=1-areaMin2

        fig2 = plt.figure(figsize=(12,8))
        ax = fig2.add_subplot(111)
#        ax.patch.set_facecolor('white')
#        plt.grid(True,linestyle="-", color=[0.7,0.7,0.7])
        ax.plot( [30], [0.09], '.', color='white', label='1-P ='+str(Pkval[mm])[:4])
        ax.plot( centerX, testM2, 'k--',linewidth=3,label="EPDF common area")
        ax.bar(centerX, histX, align='center', width=widthX, fc='blue',alpha=0.7,label="EPDF of observations")
        ax.bar(centerY, histY, align='center', width=2*widthY/3, fc='red',alpha=0.6,label="EPDF of "+newFrame.columns[mm])

        plt.xlabel("RX1day (mm/day)", fontsize=24)
        plt.ylabel("EPDF", fontsize=24)
        plt.yticks(fontsize=24)
        plt.xticks(fontsize=24)
        plt.xticks([-40,-20,0,20,40,60,80,100],fontsize=24)
        plt.title('c)' ,fontsize=24)

        lg=plt.legend(loc=1, fontsize=18)
        lg.draw_frame(False)

        plt.savefig(output+'Fig9c.png')

        fig2.clf()

    print ('OK Kuiper Perkins')


