from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

def main():

    # odeint as a function of HIF_0 and O2_0
    def model2(y, t, HIF_0, O2_0):
        kmax_o2_deg = 1e-2
        HIF_threshold = 2.5
        hill_coeff = 10

        HIF = y[0]
        O2 = y[1]

        dHIFdt = 0
        dOdt = 0
        if O2 > 0:
            dOdt = - kmax_o2_deg / ((HIF/HIF_threshold)**hill_coeff + 1)

        return [dHIFdt, dOdt]

    # odeint
    HIF_0 = 1
    O2_0 = 1
    t = np.linspace(0, 200)
    y = odeint(model2, [HIF_0, O2_0], t, args=(HIF_0, O2_0))

    print(f'RESULTS: {y}')

    # plot results
    plt.plot(t, y)
    plt.xlabel('time')
    plt.legend(['HIF', 'O2'])
    # plt.ylabel('HIF')
    plt.show()

    # plot how HIF affects O2 consumption
    O2_results = []

    # for HIF from 0 to 1 with step 0.01
    t2 = np.linspace(0, 5, 100)
    for HIF in t2:
        y = model2([HIF, 1], t, HIF, 1)
        O2_results.append(y[1])

    plt.plot(t2, O2_results)
    plt.xlabel('HIF')
    plt.ylabel('dO2/dt')
    plt.show()




if __name__ == '__main__':
    main()