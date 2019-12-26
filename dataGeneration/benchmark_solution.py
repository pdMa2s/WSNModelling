import math
import matplotlib.pyplot as plt
from scipy.optimize import BFGS
from scipy.optimize import Bounds
from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize


# Cálculo dos consumos em cada t
def Caudal_VC(ti, tf):
    # definição do polinómio
    a6 = -5.72800E-05
    a5 = 3.9382E-03
    a4 = -9.8402E-02
    a3 = 1.0477
    a2 = -3.8621
    a1 = -1.1695
    a0 = 7.53930E+01
    QVC = a6/7.*(tf**7.-ti**7.)+a5/6.*(tf**6.-ti**6.)+a4/5.*(tf**5.-ti**5.)+ a3/4.*(tf**4.-ti**4.)+a2/3.*(tf**3.-ti**3.)+a1/2.*(tf**2.-ti**2.)+a0*(tf-ti)
    return QVC


def Caudal_R(ti, tf):
    # definição do polinómio
    a3 = -0.004
    a2 = 0.09
    a1 = 0.1335
    a0 = 20.0
    QR = a3/4.*(tf**4.-ti**4.)+a2/3.*(tf**3.-ti**3.)+a1/2.*(tf**2.-ti**2.)+a0*(tf-ti)
    return QR


def get_demand(data, t):
    return data[t]


def demand_flows(flow_r, flow_vc, start_time, end_time=None, data_r=None, data_vc=None):
    d_r = -1
    d_vc = -1
    if flow_r:
        d_r = flow_r(data_r, int(start_time))
    else:
        d_r = Caudal_R(start_time, end_time)

    if flow_vc:
        d_vc = flow_vc(data_vc, int(start_time))
    else:
        d_vc = Caudal_VC(start_time, end_time)

    return d_r, d_vc


def tarifario(ti):
    # definição do tarifário usando o tempo inicial do incremento
    tarifHora = [None]*7
    tarifCusto = [None]*7
    set(tarifHora)
    tarifHora[0] = 0; tarifCusto[0] = 0.0737
    tarifHora[1] = 2; tarifCusto[1] = 0.06618
    tarifHora[2] = 6; tarifCusto[2] = 0.0737
    tarifHora[3] = 7; tarifCusto[3] = 0.10094
    tarifHora[4] = 9; tarifCusto[4] = 0.18581
    tarifHora[5] = 12; tarifCusto[5] = 0.10094
    tarifHora[6] = 24; tarifCusto[6] = 0.10094
    tarifF = 0.
    for i in range(0, len(tarifHora)-1):
        if (ti >= tarifHora[i]) & (ti < tarifHora[i+1]):
            tarifF = tarifCusto[i]
            break
    if tarifF == 0.:
        print("Erro no tarifário", ti, i)
        quit()
    return tarifF


def benchmark2018(x, iChart, step=3600, hF0=4, flow_r=None, flow_vc=None, demand_r=None, demand_vc=None):
    nInc = len(x)
    # definição dos dicionários
    empty_timeIncrem = {
        'number': None,
        'startTime': None, 'duration': None, 'endTime': None,
        'hIni': [], 'hFin': [], 'E': -1, 'dmds': [], 'pumps': -1}
    fObjRest = {'fObj': None, 'g1': [], 'g2': []}
    Sensibil = {'dCdx': [], 'dg1dx': [[0 for j in range(nInc)]for i in range(nInc)],
                'dg2dx': [[0 for j in range(nInc)]for i in range(nInc)]}

    # Dados gerais
    hmin = 2
    hmax = 7
    hFixo = 100
    AF = 155.0
    V0 = 620.0
    deltahF = 0

    g = 9.81; densidade = 1000.0
    LPR = 3500; LRF = 6000
    f = 0.02; d = 0.3

    # Dados da bomba
    a1 = 280.; a2 = -0.0027; etaP = 0.75
    # variáveis constantes
    f32gpi2d5 = 32.0*f/(g*math.pi**2.0*d**5.)
    aRes = (a2*step**2.) - f32gpi2d5*LPR - f32gpi2d5*LRF

    # Inicialização dos vetores
    timeInc = []
    CustoT = 0
    for i in range(0, nInc):
        # definição dos incrementos de tempo
        timeInc.append(empty_timeIncrem.copy())
        timeInc[i]['number'] = i + 1
        if i == 0:
            timeInc[i]['startTime'] = 0
            hF = hF0
            timeInc[i]['hIni'] = hF
        else:
            timeInc[i]['startTime'] = timeInc[i-1]['endTime']
            timeInc[i]['hIni'] = timeInc[i-1]['hFin']
        timeInc[i]['duration'] = 24 / nInc
        timeInc[i]['endTime'] = timeInc[i]['startTime']+timeInc[i]['duration']
        #print ("timeInc", timeInc[i]['number'],timeInc[i]['startTime'],timeInc[i]['duration'])
        #
        # Cálculo dos volumes bombeados no incremento i
        QR, QVC = demand_flows(flow_r, flow_vc, timeInc[i]['startTime'], timeInc[i]['endTime'], demand_r, demand_vc)
        QRmed= QR/timeInc[i]['duration']
        timeInc[i]['dmds'] = [QVC, QR]
        #
        # Ciclo iterativo de convergência (com tolerãncia=1.E-5)
        iter = 1
        hFini = hF
        hFmed = hF
        deltahFold = 0.
        tol = 1.E-6
        maxIter = 8
        bRes = 2.*f32gpi2d5*LRF*QRmed/step
        while iter < maxIter:
            cRes = a1-hFixo -f32gpi2d5*LRF*(QRmed/step)**2 - hFmed
            Qp = (-bRes - math.sqrt(bRes**2 - 4 * aRes * cRes))/(2*aRes) * step
            deltahFn = (Qp*x[i]*timeInc[i]['duration']-QVC-QR)/AF
            hF = hFini + deltahFn
            hFmed = hFini + deltahFn / 2
            #print("iter=",iter,cRes, Qp, deltahFn,deltahFold, hF, deltahFn-deltahFold)
            if math.fabs(deltahFn-deltahFold) > tol:
                deltahFold = deltahFn
            else:
                break
            iter += 1
        timeInc[i]['hFin'] = hF
        #
        # Cálculo da energia utilizada
        WP = g*densidade/etaP*Qp/step*(a1+a2*Qp**2.)    # in W
        tarifInc = tarifario(timeInc[i]['startTime'])*timeInc[i]['duration']/1000.# in Euro/W
        Custo = x[i] * WP * tarifInc
        w_spent = (x[i] * WP)/1000
        CustoT += Custo
        timeInc[i]['E'] = w_spent
        timeInc[i]['pumps'] = x[i]
        fObjRest['g1'].append(hmin - hF)
        fObjRest['g2'].append(hF - hmax)
        #print("it.= %2i, x= %5.3f, hF= %6.3f, WP= %7.3f, Tarif= %5.3f, Custo= %6.3f, %7.3f, constr= %7.3f, %7.3f, <0 ?"
        #      % (i, x[i], hF, WP, tarifario(timeInc[i]['startTime']), Custo, CustoT, fObjRest['g1'][i],fObjRest['g2'][i]))

        # Cálculo de sensibilidades (aproximadas, pois consideram que Qp é independente de x)
        Sensibil['dCdx'].append(WP*tarifInc)
        dgP = Qp*timeInc[i]['duration']/(AF + 0.5*x[i]*timeInc[i]['duration']*(bRes**2-4.*aRes*cRes)**(-0.5))
        #
        # ciclo para cada dg1dx[i][j], onde i=alfa do x; j=inc. e j>=alfa
        for j in range(i, nInc):
            Sensibil['dg1dx'][i][j] = -dgP
            Sensibil['dg2dx'][i][j] = dgP
    # Guardar valores em Arrays
    fObjRest['fObj'] = CustoT

    # Construção da solução grafica
    #iChart = 0
    if iChart == 1:
        x1 = []
        y1 = []
        z1 = []
        for i in range(0, nInc):
            x1.insert(i, timeInc[i]['startTime'])
            y1.insert(i, timeInc[i]['hIni'])
            z1.insert(i, 10*tarifario(i/(nInc/24)))
        x1.insert(nInc, timeInc[nInc-1]['endTime'])
        y1.insert(nInc, timeInc[nInc-1]['hFin'])
        z1.insert(nInc, 10*tarifario(i/(nInc/24)))
        plt.plot(x1, y1, x1[0:nInc], x[0:nInc], x1, z1)
        plt.title('Solução Proposta, Custo=%f' % CustoT)
        plt.xlabel('Tempo (h)')
        plt.ylabel('Nivel/ status da bomba / Tarifario (x10)')
        plt.grid()
        plt.show()

    #print ("end benchmark 2018")
    return fObjRest, Sensibil, timeInc


def fun_obj(x):
    res, sens, time = benchmark2018(x, 0)
    cost = res['fObj']
    return cost


def fun_constr_1(x):
    res, sens, time = benchmark2018(x, 0)
    g1 = res['g1']
    return g1


def fun_constr_2(x):
    res, sens, time = benchmark2018(x, 0)
    g2 = res['g2']
    return g2


if __name__ == '__main__':

    # main program (driver)
    # ----------------------
    nInc = 96 #24
    # Declaração de solução
    x = [0.5 for i in range(0, nInc)]
    # pq esta declaração?
    # fObjRest, Sensibil, timeInc = Benchmark2018(x, 1)

    c1 = NonlinearConstraint(fun_constr_1, -9999999, 0, jac='2-point', hess=BFGS(), keep_feasible=False)
    c2 = NonlinearConstraint(fun_constr_2, -9999999, 0, jac='2-point', hess=BFGS(), keep_feasible=False)
    bounds = Bounds([0 for i in range(nInc)], [1 for i in range(nInc)], keep_feasible=False)
    #res = minimize(fun_obj, x, args=(), method='trust-constr', jac='2-point', hess=BFGS(), constraints=[c1, c2],
     #              options={'verbose': 3}, bounds=bounds)
    #print("res=",res)
    #print("Solução final: x=", [round(res.x[i], 3) for i in range(len(res.x))])
    #a=input('')

    fObjRest, sensibil, time_inc = benchmark2018(x, 1, step=3600)
    print("CustoF=", fObjRest['fObj'], '\n')


