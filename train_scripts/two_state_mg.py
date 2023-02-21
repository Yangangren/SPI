import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import pi
from matplotlib import rcParams
from matplotlib.pyplot import MultipleLocator
rho_list = [1.0, 5.0, 10.0, 20.0]



def policy_iteration():
    ############################################### 1st iteration ##########################################
    v_1, v_2 = 0., 0.
    v_1_appr, v_2_appr = 0., 0.
    pi_1, pi_2 = 0.5, 0.5   # (API, SPI) 0.5, 0.5; (NPI) 1.0, 0.   # todo
    mu_1, mu_2 = 0.45, 0.55   # (API, SPI) 0.5, 0.5; (NPI) 1.0, 0.   # todo
    gamma = 3 / 4

    value_true, value_appr, value_appr_u, value_newt = [0.], [[0.] for i in range(len(rho_list))], [0.], [0.]
    for iter in range(20):
        # worst-case PEV
        v_u1 = pi_1 * (3.0 + gamma * v_1) + pi_2 * (2.0 + gamma * v_1)  # take (a1, u1) + (a2, u1)
        v_u2 = pi_1 * (1 / 3 * (6.0 + gamma * v_1) + 2 / 3 * (6.0 + gamma * v_2)) + pi_2 * (1.0 + gamma * v_1)  # take (a1, u2) + (a2, u2)

        v_1 = min(v_u1, v_u2)
        value_true.append(v_1)

    v_1, v_2 = 0., 0.
    for iter in range(20):
        v_u1 = mu_1 * pi_1 * (3.0 + gamma * v_1) + mu_1 * pi_2 * (2.0 + gamma * v_1)                                               # take (a1, u1) + (a2, u1)
        v_u2 = mu_2 * pi_1 * (1/3 * (6.0 + gamma * v_1) + 2/3 * (6.0 + gamma * v_2)) + mu_2 * pi_2 * (1.0 + gamma * v_1)           # take (a1, u2) + (a2, u2)

        v_1 = v_u1 + v_u2
        value_newt.append(v_1)

    # SPI
    for i, rho in enumerate(rho_list):
        v_1_appr, v_2_appr = 0., 0.
        for iter in range(20):
            v_u1 = pi_1 * (3.0 + gamma * v_1_appr) + pi_2 * (2.0 + gamma * v_1_appr)  # take (a1, u1) + (a2, u1)
            v_u2 = pi_1 * (1 / 3 * (6.0 + gamma * v_1_appr) + 2 / 3 * (6.0 + gamma * v_2_appr)) + pi_2 * (1.0 + gamma * v_1_appr)  # take (a1, u2) + (a2, u2)

            v_1_appr = - 1 / rho * np.log(np.sum(np.array([mu_1, mu_2]) * np.exp(np.array([v_u1, v_u2]) * -rho)))

            value_appr[i].append(v_1_appr)

    # SPI-uniform (only for rho=10)
    for i, rho in enumerate([10]):
        v_1_appr, v_2_appr = 0., 0.
        for iter in range(20):
            v_u1 = pi_1 * (3.0 + gamma * v_1_appr) + pi_2 * (2.0 + gamma * v_1_appr)  # take (a1, u1) + (a2, u1)
            v_u2 = pi_1 * (1 / 3 * (6.0 + gamma * v_1_appr) + 2 / 3 * (6.0 + gamma * v_2_appr)) + pi_2 * (1.0 + gamma * v_1_appr)  # take (a1, u2) + (a2, u2)

            v_1_appr = - 1 / rho * np.log(np.sum(np.array([0.5, 0.5]) * np.exp(np.array([v_u1, v_u2]) * -rho)))
            value_appr_u.append(v_1_appr)

    time_line = np.array([k for k in range(len(value_true))])
    color_list = ['b', 'cyan', 'g', 'orange', 'r']
    color1, color2, color3, color4 = 'k', 'r', 'b', 'g'
    style1, style2, style3, style4 = '-', '--', '-.', ':'
    linewidth = 1.0

    f = plt.figure(1, figsize=(6.0, 4.4))
    ax = plt.axes([0.10, 0.12, 0.85, 0.85])
    ax.plot(time_line, -1 * np.array(value_true), linewidth=linewidth, color=color1, linestyle=style1)
    for i in range(len(rho_list)):
        ax.plot(time_line, -1 * np.array(value_appr[i]), linewidth=linewidth, color=color_list[i], linestyle=style2)
    ax.plot(time_line, -1 * np.array(value_appr_u), linewidth=linewidth, color=color_list[-1], linestyle=style4)
    # ax.plot(time_line, -1 * np.array(value_newt), linewidth=linewidth, color=color1, linestyle=style3)
    x_locater = MultipleLocator(2)
    ax.xaxis.set_major_locator(x_locater)
    # ax.set_ylabel('值函数')
    # ax.set_xlabel("迭代次数")
    ax.set_xlim([0, 20])
    ax.set_ylim([-8.5, -2])
    ax.set_ylabel('Value', fontsize=14)
    ax.set_xlabel("PEV Step", fontsize=14)
    # ax.legend(frameon=False, fontsize=20)
    plt.legend(['API', r'SPI-a($\rho=1.0$)', r'SPI-a($\rho=5.0$)', r'SPI-a($\rho=10.0$)', r'SPI-a($\rho=20.0$)', r'SPI-u($\rho=10.0$)'], frameon=False)
    f_name = 'pev-iter1'
    plt.savefig('./{}.pdf'.format(f_name))
    plt.close(f)

    # PIM
    matrix_true = np.array([[3.0 + gamma * v_1, 1 / 3 * (6.0 + gamma * v_1) + 2 / 3 * (6.0 + gamma * v_2)],
                            [2.0 + gamma * v_1, 1.0 + gamma * v_1]])
    matrix_appr = np.array([[3.0 + gamma * v_1_appr, 1 / 3 * (6.0 + gamma * v_1_appr) + 2 / 3 * (6.0 + gamma * v_2_appr)],
                            [2.0 + gamma * v_1_appr, 1.0 + gamma * v_1_appr]])
    print("matrix_true1 \n", matrix_true)
    print("matrix_appr1 \n", matrix_appr)

    ############################################### 2nd iteration ############################################
    v_1, v_2 = 0., 0.
    v_1_appr, v_2_appr = 0., 0.
    pi_1, pi_2 = 1.0, 0.0            # (API, SPI) 1.0, 0.0; (NPI) 1.0, 1.0   # todo
    mu_1, mu_2 = 0.0, 1.0            # (API, SPI) 0.0, 1.0; (NPI) 0.0, 1.0   # todo
    gamma = 3 / 4
    value_true, value_appr, value_appr_u, value_newt = [0.], [[0.] for i in range(len(rho_list))], [0.], [0.]

    for iter in range(20):
        v_u1 = pi_1 * (3.0 + gamma * v_1) + pi_2 * (2.0 + gamma * v_1)  # take (a1, u1) + (a2, u1)
        v_u2 = pi_1 * (1 / 3 * (6.0 + gamma * v_1) + 2 / 3 * (6.0 + gamma * v_2)) + pi_2 * (1.0 + gamma * v_1)  # take (a1, u2) + (a2, u2)

        v_1 = min(v_u1, v_u2)
        value_true.append(v_1)

    v_1, v_2 = 0., 0.
    for iter in range(20):
        v_u1 = mu_1 * pi_1 * (3.0 + gamma * v_1) + mu_1 * pi_2 * (2.0 + gamma * v_1)                                               # take (a1, u1) + (a2, u1)
        v_u2 = mu_2 * pi_1 * (1/3 * (6.0 + gamma * v_1) + 2/3 * (6.0 + gamma * v_2)) + mu_2 * pi_2 * (1.0 + gamma * v_1)           # take (a1, u2) + (a2, u2)

        v_1 = v_u1 + v_u2
        value_newt.append(v_1)

    # SPI
    for i, rho in enumerate(rho_list):
        v_1_appr, v_2_appr = 0., 0.
        for iter in range(20):
            v_u1 = pi_1 * (3.0 + gamma * v_1_appr) + pi_2 * (2.0 + gamma * v_1_appr)  # take (a1, u1) + (a2, u1)
            v_u2 = pi_1 * (1 / 3 * (6.0 + gamma * v_1_appr) + 2 / 3 * (6.0 + gamma * v_2_appr)) + pi_2 * (1.0 + gamma * v_1_appr)  # take (a1, u2) + (a2, u2)

            v_1_appr = - 1 / rho * np.log(np.sum(np.array([mu_1, mu_2]) * np.exp(np.array([v_u1, v_u2]) * -rho)))

            value_appr[i].append(v_1_appr)

    # SPI-uniform (only for rho=20)
    for i, rho in enumerate([10]):
        v_1_appr, v_2_appr = 0., 0.
        for iter in range(20):
            v_u1 = pi_1 * (3.0 + gamma * v_1_appr) + pi_2 * (2.0 + gamma * v_1_appr)  # take (a1, u1) + (a2, u1)
            v_u2 = pi_1 * (1 / 3 * (6.0 + gamma * v_1_appr) + 2 / 3 * (6.0 + gamma * v_2_appr)) + pi_2 * (1.0 + gamma * v_1_appr)  # take (a1, u2) + (a2, u2)

            v_1_appr = - 1 / rho * np.log(np.sum(np.array([0.5, 0.5]) * np.exp(np.array([v_u1, v_u2]) * -rho)))
            value_appr_u.append(v_1_appr)

    time_line = np.array([k for k in range(len(value_true))])
    f2 = plt.figure(1, figsize=(6.0, 4.4))
    ax2 = plt.axes([0.10, 0.12, 0.85, 0.85])
    ax2.plot(time_line, -1 * np.array(value_true), linewidth=linewidth, color=color1, linestyle=style1)
    for i in range(len(rho_list)):
        ax2.plot(time_line, -1 * np.array(value_appr[i]), linewidth=linewidth, color=color_list[i], linestyle=style2)
    ax2.plot(time_line, -1 * np.array(value_appr_u), linewidth=linewidth, color=color_list[-1], linestyle=style4)
    # ax2.plot(time_line, -1 * np.array(value_newt), linewidth=linewidth, color=color1, linestyle=style3)
    x_locater = MultipleLocator(2)
    ax2.xaxis.set_major_locator(x_locater)
    # ax2.set_ylabel('值函数')
    # ax2.set_xlabel("迭代次数")
    ax2.set_xlim([0, 20])
    ax2.set_ylim([-8.5, -2])
    ax2.set_ylabel('Value', fontsize=14)
    ax2.set_xlabel("PEV Step", fontsize=14)
    # ax.legend(frameon=False, fontsize=20)
    # plt.legend(['API', r'SPI($\rho=1.0$)', r'SPI($\rho=5.0$)', r'SPI($\rho=10.0$)', r'SPI($\rho=20.0$)'], frameon=False)
    f_name = 'pev-iter2'
    plt.savefig('./{}.pdf'.format(f_name))
    plt.close(f2)

    # PIM
    matrix_true = np.array([[3.0 + gamma * v_1, 1 / 3 * (6.0 + gamma * v_1) + 2 / 3 * (6.0 + gamma * v_2)],
                            [2.0 + gamma * v_1, 1.0 + gamma * v_1]])
    matrix_appr = np.array([[3.0 + gamma * v_1_appr, 1 / 3 * (6.0 + gamma * v_1_appr) + 2 / 3 * (6.0 + gamma * v_2_appr)],
                            [2.0 + gamma * v_1_appr, 1.0 + gamma * v_1_appr]])
    print("matrix_true2 \n ", matrix_true)
    print("matrix_appr2 \n", matrix_appr)

    ########################################### 3rd iteration ######################################################
    v_1, v_2 = 0., 0.
    v_1_appr, v_2_appr = 0., 0.
    pi_1, pi_2 = 1.0, 0.
    mu_1, mu_2 = 0., 1.0
    gamma = 3 / 4
    value_true, value_appr, value_appr_u, value_newt = [0.], [[0.] for i in range(len(rho_list))], [0.], [0.]

    for iter in range(20):
        v_u1 = pi_1 * (3.0 + gamma * v_1) + pi_2 * (2.0 + gamma * v_1)  # take (a1, u1) + (a2, u1)
        v_u2 = pi_1 * (1 / 3 * (6.0 + gamma * v_1) + 2 / 3 * (6.0 + gamma * v_2)) + pi_2 * (1.0 + gamma * v_1)  # take (a1, u2) + (a2, u2)

        v_1 = min(v_u1, v_u2)
        value_true.append(v_1)

    # SPI
    for i, rho in enumerate(rho_list):
        v_1_appr, v_2_appr = 0., 0.
        for iter in range(20):
            v_u1 = pi_1 * (3.0 + gamma * v_1_appr) + pi_2 * (2.0 + gamma * v_1_appr)  # take (a1, u1) + (a2, u1)
            v_u2 = pi_1 * (1 / 3 * (6.0 + gamma * v_1_appr) + 2 / 3 * (6.0 + gamma * v_2_appr)) + pi_2 * (1.0 + gamma * v_1_appr)  # take (a1, u2) + (a2, u2)

            v_1_appr = - 1 / rho * np.log(np.sum(np.array([mu_1, mu_2]) * np.exp(np.array([v_u1, v_u2]) * -rho)))

            value_appr[i].append(v_1_appr)

    # SPI-uniform (only for rho=10)
    for i, rho in enumerate([10]):
        v_1_appr, v_2_appr = 0., 0.
        for iter in range(20):
            v_u1 = pi_1 * (3.0 + gamma * v_1_appr) + pi_2 * (2.0 + gamma * v_1_appr)  # take (a1, u1) + (a2, u1)
            v_u2 = pi_1 * (1 / 3 * (6.0 + gamma * v_1_appr) + 2 / 3 * (6.0 + gamma * v_2_appr)) + pi_2 * (1.0 + gamma * v_1_appr)  # take (a1, u2) + (a2, u2)

            v_1_appr = - 1 / rho * np.log(np.sum(np.array([0.5, 0.5]) * np.exp(np.array([v_u1, v_u2]) * -rho)))
            value_appr_u.append(v_1_appr)

    time_line = np.array([k for k in range(len(value_true))])

    f3 = plt.figure(1, figsize=(6.0, 4.4))
    ax3 = plt.axes([0.10, 0.12, 0.85, 0.85])
    ax3.plot(time_line, -1 * np.array(value_true), linewidth=linewidth, color=color1, linestyle=style1)
    for i in range(len(rho_list)):
        ax3.plot(time_line, -1 * np.array(value_appr[i]), linewidth=linewidth, color=color_list[i], linestyle=style2)
    ax3.plot(time_line, -1 * np.array(value_appr_u), linewidth=linewidth, color=color_list[-1], linestyle=style4)
    x_locater = MultipleLocator(2)
    ax3.xaxis.set_major_locator(x_locater)
    # ax3.set_ylabel('值函数')
    # ax3.set_xlabel("迭代次数")
    ax3.set_xlim([0, 20])
    ax3.set_ylim([-8.5, -2])
    ax3.set_ylabel('Value', fontsize=14)
    ax3.set_xlabel("PEV Step", fontsize=14)
    # ax.legend(frameon=False, fontsize=20)
    # plt.legend(['API', r'SPI($\rho=1.0$)', r'SPI($\rho=5.0$)', r'SPI($\rho=10.0$)', r'SPI($\rho=20.0$)'], frameon=False)
    f_name = 'pev-iter3'
    plt.savefig('./{}.pdf'.format(f_name))
    plt.close(f3)

    # PIM
    matrix_true = np.array([[3.0 + gamma * v_1, 1 / 3 * (6.0 + gamma * v_1) + 2 / 3 * (6.0 + gamma * v_2)],
                            [2.0 + gamma * v_1, 1.0 + gamma * v_1]])
    matrix_appr = np.array([[3.0 + gamma * v_1_appr, 1 / 3 * (6.0 + gamma * v_1_appr) + 2 / 3 * (6.0 + gamma * v_2_appr)],
                            [2.0 + gamma * v_1_appr, 1.0 + gamma * v_1_appr]])
    print("matrix_true3 \n", matrix_true)
    print("matrix_appr3 \n", matrix_appr)

    # ################################### Naive policy iteration #######################################################
    # ################################### 1st iteration ################################################################
    print("------------------------------------Naive policy iteration ------------------------------------------------")
    v_1, v_2 = 0., 0.
    pi_1, pi_2 = 1.0, 0.0                          # 0.5, 0.5
    mu_1, mu_2 = 1.0, 0.0                          # 0.5, 0.5
    gamma = 3/4

    matrix_true = np.array([[3.0 + gamma * v_1, 1 / 3 * (6.0 + gamma * v_1) + 2 / 3 * (6.0 + gamma * v_2)],
                            [2.0 + gamma * v_1, 1.0 + gamma * v_1]])
    print("The init matrix: \n", matrix_true)

    for iter in range(50):
        v_u1 = mu_1 * pi_1 * (3.0 + gamma * v_1) + mu_1 * pi_2 * (2.0 + gamma * v_1)                                               # take (a1, u1) + (a2, u1)
        v_u2 = mu_2 * pi_1 * (1/3 * (6.0 + gamma * v_1) + 2/3 * (6.0 + gamma * v_2)) + mu_2 * pi_2 * (1.0 + gamma * v_1)           # take (a1, u2) + (a2, u2)

        v_1 = v_u1 + v_u2

    matrix_true = np.array([[3.0 + gamma * v_1, 1 / 3 * (6.0 + gamma * v_1) + 2 / 3 * (6.0 + gamma * v_2)],
                            [2.0 + gamma * v_1, 1.0 + gamma * v_1]])
    print("matrix_true1 \n", matrix_true)

    ################################### 2nd iteration ################################################################
    v_1, v_2 = 0., 0.
    pi_1, pi_2 = 0.0, 1.0
    mu_1, mu_2 = 0.0, 1.0
    gamma = 3/4

    for iter in range(50):
        v_u1 = mu_1 * pi_1 * (3.0 + gamma * v_1) + mu_1 * pi_2 * (2.0 + gamma * v_1)                                               # take (a1, u1) + (a2, u1)
        v_u2 = mu_2 * pi_1 * (1/3 * (6.0 + gamma * v_1) + 2/3 * (6.0 + gamma * v_2)) + mu_2 * pi_2 * (1.0 + gamma * v_1)           # take (a1, u2) + (a2, u2)

        v_1 = v_u1 + v_u2

    matrix_true = np.array([[3.0 + gamma * v_1, 1 / 3 * (6.0 + gamma * v_1) + 2 / 3 * (6.0 + gamma * v_2)],
                            [2.0 + gamma * v_1, 1.0 + gamma * v_1]])
    print("matrix_true2 \n", matrix_true)

    ################################### 3rd iteration ################################################################
    v_1, v_2 = 12., 0.
    pi_1, pi_2 = 1.0, 0.0
    mu_1, mu_2 = 1.0, 0.0
    gamma = 3/4

    for iter in range(50):
        v_u1 = mu_1 * pi_1 * (3.0 + gamma * v_1) + mu_1 * pi_2 * (2.0 + gamma * v_1)                                               # take (a1, u1) + (a2, u1)
        v_u2 = mu_2 * pi_1 * (1/3 * (6.0 + gamma * v_1) + 2/3 * (6.0 + gamma * v_2)) + mu_2 * pi_2 * (1.0 + gamma * v_1)           # take (a1, u2) + (a2, u2)

        v_1 = v_u1 + v_u2

    matrix_true = np.array([[3.0 + gamma * v_1, 1 / 3 * (6.0 + gamma * v_1) + 2 / 3 * (6.0 + gamma * v_2)],
                            [2.0 + gamma * v_1, 1.0 + gamma * v_1]])
    print("matrix_true3 \n", matrix_true)

    ################################### 4rd iteration ################################################################
    v_1, v_2 = 12., 0.
    pi_1, pi_2 = 0.0, 1.0
    mu_1, mu_2 = 0.0, 1.0
    gamma = 3/4

    for iter in range(50):
        v_u1 = mu_1 * pi_1 * (3.0 + gamma * v_1) + mu_1 * pi_2 * (2.0 + gamma * v_1)                                               # take (a1, u1) + (a2, u1)
        v_u2 = mu_2 * pi_1 * (1/3 * (6.0 + gamma * v_1) + 2/3 * (6.0 + gamma * v_2)) + mu_2 * pi_2 * (1.0 + gamma * v_1)           # take (a1, u2) + (a2, u2)

        v_1 = v_u1 + v_u2

    matrix_true = np.array([[3.0 + gamma * v_1, 1 / 3 * (6.0 + gamma * v_1) + 2 / 3 * (6.0 + gamma * v_2)],
                            [2.0 + gamma * v_1, 1.0 + gamma * v_1]])
    print("matrix_true3 \n", matrix_true)

    ############################### NPI Iteration #####################################################################
    npi = [-12.0, 4.0]
    npi_list = [-12.0, -4.0] * 10

    time_line = np.array([k for k in range(len(npi_list))])
    f5 = plt.figure(1, figsize=(6.0, 4.4))
    ax5 = plt.axes([0.11, 0.12, 0.88, 0.85])
    ax5.plot(time_line, np.array(npi_list), marker='o', markersize=8, markerfacecolor='white', markeredgecolor='black', markeredgewidth=1,
             linewidth=1, color=color1, linestyle=style2)
    x_locater = MultipleLocator(2)
    ax5.xaxis.set_major_locator(x_locater)
    # ax2.set_ylabel('值函数')
    # ax2.set_xlabel("迭代次数")
    ax5.set_ylabel('Value', fontsize=16)
    ax5.set_xlabel("NPI Iteration", fontsize=16)
    # ax.legend(frameon=False, fontsize=20)
    # plt.legend(['API', r'SPI($\rho=1.0$)', r'SPI($\rho=5.0$)', r'SPI($\rho=10.0$)', r'SPI($\rho=20.0$)'], frameon=False)
    f_name = 'npi-iter'
    plt.savefig('./{}.pdf'.format(f_name))
    plt.close(f2)

if __name__ == "__main__":
    policy_iteration()