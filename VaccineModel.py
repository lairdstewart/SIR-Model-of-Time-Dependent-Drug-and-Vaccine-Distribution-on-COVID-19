import numpy as np
import matplotlib.pyplot as plt  # plotting
import scipy
from scipy.integrate import odeint  # ODE integrator
import matplotlib.dates as dt  # module for dealing with dates
from scipy.optimize import curve_fit
from scipy.integrate import quad


# ======================================================================================================================
def vaccine_sir(init_cond, t, Ba, Bi, p, m, ni, nr, g, a, k, a_log, b_log, n_log):
    """
    :param b: (beta) infectivity rate
    :param p: (rho) proportion of infected people DYING per day
    :param m: (mu) percentage of asymptomatic people that get sick
    :param ni: how long someone is asymptomatic before becoming infectious (1 / num days)
    :param nr: how long someone is asymptomatic if they never become infectious (1 / num days)
    :param g: (gamma) percentage of infected people recovering per day (1 / recovery days)
    :param a: (alpha) percentage of people that die from the virus
    :paran v: how many people are vaccinated that day
    :param k: (kappa) the effectiveness of the vaccine ie. 9/10 means for 9 of 10 people it is effective
    """

    v = logistic_distribution(t, a_log, b_log, n_log)        # logistic vaccines release
    # v = 0

    S, Siv, A, I, R, D = init_cond

    if v < S - Bi*I*(S/N) - Ba*A*(S/N):  # used to just be v < S DELETE THIS
        dS = - Bi*I*(S/N) - Ba*A*(S/N) - k*v - (1-k)*v
    else:
        dS = - k*S - (1-k)*S

    if v < S - (Bi*I*(S/N) + Ba*A*(S/N)):
        dSiv = + (1-k)*v - Bi*I*(Siv/N) - Ba*A*(Siv/N)
    else:
        dSiv = + (1-k)*S - Bi*I*(Siv/N) - Ba*A*(Siv/N)

    if v < S - (Bi*I*(S/N) + Ba*A*(S/N)):
        dA = + Bi*I*(S/N) + Ba*A*(S/N) + Bi*I*(Siv/N) + Ba*A*(Siv/N) - m*A*ni - (1-m)*A*nr
    else:
        dA = + Bi*I*(Siv/N) + Ba*A*(Siv/N) - m*A*ni - (1-m)*A*nr

    dI = + m*A*ni - a*p*I - (1-a)*g*I

    if v < S - (Bi*I*(S/N) + Ba*A*(S/N)):
        dR = + k*v + (1-m)*A*nr + (1-a)*g*I
    else:
        dR = + k*S + (1-m)*A*nr + (1-a)*g*I

    dD = + a*p*I

    return np.array((dS, dSiv, dA, dI, dR, dD))


def integrator(t, init_cond, Ba, Bi, p, m, ni, nr, g, a, k, a_log, b_log, n_log):
    output = odeint(vaccine_sir, init_cond, t, args=(Ba, Bi, p, m, ni, nr, g, a, k, a_log, b_log, n_log))
    S = output[:, 0]
    Siv = output[:, 1]
    A = output[:, 2]
    I = output[:, 3]
    R = output[:, 4]
    D = output[:, 5]
    return np.array((S, Siv, A, I, R, D))


def logistic_distribution(x, a, b, total_num_vaccines):
    n = total_num_vaccines
    if x == 0:  #
        pass
    return n*a*np.exp(a*(b-x))/(1+np.exp(a*(b-x)))**2


def logistic_function(x, a, b, total_num_vaccines):
    n = total_num_vaccines
    return n/(1+np.exp(a*(b-x)))


# ======================================================================================================================
# VARIABLES
SimulationTime = 400.0
dt = 1
t = np.arange(0.0, SimulationTime + dt, dt)

# SAIRD Model Variables
Ba = 0.120527   # infectivity of asymptomatic people
Bi = 0.120527   # infectivity of infected (symptomatic) people
mu = 0.85       # percentage of asymptomatic people that get sick
nu_i = 0.1961   # proportion of people transitioning from asymptomatic to infected
nu_r = 0.1111   # proportion of people transitioning from asymptomatic to recovered
p = 0.0562      # proportion of people dying per day (no drug). aka. it takes 7 days to die
g = 0.0870      # proportion of people recovering per day. aka. it takes 11 days to recover
a = 0.0064      # death rate ie. 10% of infected people will die

# Vaccine Variables
k = 0.91             # effectiveness of the vaccine
a_log = 0.1           # a in logistic distribution equation (see function)
b_log = 250         # b in logistic distribution equation (see function)
n_log = 100000000   # total number of vaccines distributed

# Starting Conditions
N = 3.28196e08      # total population of the US
I0 = 1000           # infected --> note 'infected' means infected/showing symptoms, asymptomatic are also infected
R0 = 0              # Initial Recovered Population
S0 = N - I0         # Initial Susceptible Population (N - 1)
D0 = 0              # Initial Dead Population
Siv0 = 0            # Initial Ineffectively Vaccinated Population
A0 = 0              # Initial Asymptomatic Population


if __name__ == '__main__':
    # ======================================================================================================================
    initial_conditions = np.array((S0, Siv0, A0, I0, R0, D0))
    S, Siv, A, I, R, D = integrator(t, initial_conditions, Ba, Bi, p, mu, nu_i, nu_r, g, a, k, a_log, b_log, n_log)

    # ======================================================================================================================
    # BASE MODEL PLOTTING
    print("\nFinal Conditions at t = " + str(SimulationTime) + ": ")
    print("\tSusceptible = " + str(int(S[-1])))
    print("\tSusceptible (ineffective vaccination) = " + str(int(Siv[-1])))
    print("\tAsymptomatic = " + str(int(A[-1])))
    print("\tInfected = " + str(int(I[-1])))
    print("\tRecovered = " + str(int(R[-1])))
    print("\tDead = " + str(int(D[-1])))

    plt.plot(t, S, label='Susceptible')
    plt.plot(t, Siv, label='Susceptible (innef. vaccine)')
    plt.plot(t, A, label='Asymptomatic')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.plot(t, D, label='Dead')
    plt.plot(t, N * np.ones(len(t)), label='Total Population')
    plt.xlabel('time')
    plt.ylabel('number people')
    # plt.title("SIR Model Test")
    plt.legend(loc='upper center')
    plt.legend()
    plt.show()

    peak = np.argmax(I)

    # ======================================================================================================================
    # Testing Functions
    log_curve = []          # logistic curve
    log_distribution = []   # logistic distribution curve (derivative)
    # x = np.arange(0, 501, 1)
    for i in t:
        log_curve.append(logistic_function(i, 0.1, 200, 300000000))
        log_distribution.append(logistic_distribution(i, 0.1, 200, 300000000))

    print(scipy.integrate.quad(logistic_distribution, -15, 15, args=(0.1, 0, 1)))  # --> 90.5% distributed in 30 days

    plt.plot(t, log_curve, label='Total Distribution')
    plt.plot(t, log_distribution, label='Daily Distribution (a=0.1)')
    # plt.title("Vaccine Distribution")
    plt.xlabel("Time (days)")
    plt.ylabel("Number of Vaccines")
    plt.legend()
    plt.grid()
    plt.xlim(0, 400)
    plt.show()

    log_distribution2 = []
    # testing the 'a' variable in the logistic distribution function
    x = np.arange(-40, 40, 0.01)
    for i in x:
        log_distribution2.append(logistic_distribution(i, 0.1, 0, 1))  # x, a, b, n

    plt.plot(x, log_distribution2, label='a = 0.1')
    plt.axvline(-15, color='red', alpha=0.3)
    plt.axvline(15, color='red', alpha=0.3)
    plt.xlabel("Time (days)")
    plt.ylabel("Percentage of Vaccinations Distributed")
    # plt.title("a = 0.1 --> ~63% of vaccinations occur in a one month span")
    plt.xlim(-40, 40)
    plt.ylim(0, 0.04)
    plt.legend()
    plt.show()

# ======================================================================================================================
    # DEATH VS. VACCINE RELEASE DATE (num vaccines)
    # note, we will only start releasing and end releasing the vaccine 30 days after it starts or before it ends
    release_dates = np.arange(20, 401, 0.5)
    a_log = 0.3      # sharpness of logistic distribution (constant)
    k = 0.91          # vaccine effectiveness (constant)

    # ========== total vaccines = 0.2 * N  =========
    n_log = 0.2*N    # total number of vaccines distributed
    num_vac_deaths1 = []
    for i in release_dates:
        b_log = i
        initial_conditions = np.array((S0, Siv0, A0, I0, R0, D0))
        S, Siv, A, I, R, D = integrator(t, initial_conditions, Ba, Bi, p, mu, nu_i, nu_r, g, a, k, a_log, b_log, n_log)
        num_vac_deaths1.append(D[-1])
    print("Finished # vaccines = 0.2N")

    # ========== total vaccines = 0.3 * N  =========
    n_log = 0.3*N    # total number of vaccines distributed
    num_vac_deaths2 = []
    for i in release_dates:
        b_log = i
        initial_conditions = np.array((S0, Siv0, A0, I0, R0, D0))
        S, Siv, A, I, R, D = integrator(t, initial_conditions, Ba, Bi, p, mu, nu_i, nu_r, g, a, k, a_log, b_log, n_log)
        num_vac_deaths2.append(D[-1])
    print("Finished # vaccines = 0.3N")

    # ========== total vaccines = 0.4 * N  =========
    n_log = 0.4*N    # total number of vaccines distributed per day
    num_vac_deaths3 = []
    for i in release_dates:
        b_log = i
        initial_conditions = np.array((S0, Siv0, A0, I0, R0, D0))
        S, Siv, A, I, R, D = integrator(t, initial_conditions, Ba, Bi, p, mu, nu_i, nu_r, g, a, k, a_log, b_log, n_log)
        num_vac_deaths3.append(D[-1])
    print("Finished # vaccines = 0.4N")

    # ========== total vaccines = 0.6 * N  =========
    n_log = 0.6*N    # total number of vaccines distributed
    num_vac_deaths4 = []
    for i in release_dates:
        b_log = i
        initial_conditions = np.array((S0, Siv0, A0, I0, R0, D0))
        S, Siv, A, I, R, D = integrator(t, initial_conditions, Ba, Bi, p, mu, nu_i, nu_r, g, a, k, a_log, b_log, n_log)
        num_vac_deaths4.append(D[-1])
    print("Finished # vaccines = 0.6N")

    # ========== total vaccines = 0.8 * N  =========
    n_log = 0.8*N    # total number of vaccines distributed
    num_vac_deaths5 = []
    for i in release_dates:
        b_log = i
        initial_conditions = np.array((S0, Siv0, A0, I0, R0, D0))
        S, Siv, A, I, R, D = integrator(t, initial_conditions, Ba, Bi, p, mu, nu_i, nu_r, g, a, k, a_log, b_log, n_log)
        num_vac_deaths5.append(D[-1])
    print("Finished # vaccines = 0.8N")

    plt.plot(release_dates, num_vac_deaths5, label='Total Vaccines=80% Pop.', color='black')
    plt.plot(release_dates, num_vac_deaths4, label='Total Vaccines=60% Pop.', color='xkcd:royal blue')
    plt.plot(release_dates, num_vac_deaths3, label='Total Vaccines=40% Pop.', color='xkcd:bright blue')
    plt.plot(release_dates, num_vac_deaths2, label='Total Vaccines=30% Pop.', color='xkcd:sky blue')
    plt.plot(release_dates, num_vac_deaths1, label='Total Vaccines=20% Pop.', color='silver')

    plt.axvline(peak, label='Peak Of Infections', color='red', alpha=0.3)
    # plt.title("Num. Vaccines and Total Deaths")
    plt.xlabel("Vaccine Release Date")
    plt.ylabel("Total Final Deaths")
    plt.legend()
    plt.xlim(0, 400)
    plt.show()

    # ======================================================================================================================
    # DEATH VS. VACCINE RELEASE DATE (effectiveness)
    # note, we will only start 'releasing' and end releasing the vaccine 50 days after it starts or before it ends
    release_dates = np.arange(20, 401, 0.5)
    a_log = 0.3      # see above for why i chose this value
    n_log = 0.2*N    # total number of vaccines distributed

    # ========== 20% effectiveness  =========
    k = 0.2          # vaccine effectiveness
    num_vac_deaths1 = []
    for i in release_dates:
        b_log = i
        initial_conditions = np.array((S0, Siv0, A0, I0, R0, D0))
        S, Siv, A, I, R, D = integrator(t, initial_conditions, Ba, Bi, p, mu, nu_i, nu_r, g, a, k, a_log, b_log, n_log)
        num_vac_deaths1.append(D[-1])
    print("Finished 20% effectiveness")

    # ========== 40% effectiveness  =========
    k = 0.4          # vaccine effectiveness
    num_vac_deaths2 = []
    for i in release_dates:
        b_log = i
        initial_conditions = np.array((S0, Siv0, A0, I0, R0, D0))
        S, Siv, A, I, R, D = integrator(t, initial_conditions, Ba, Bi, p, mu, nu_i, nu_r, g, a, k, a_log, b_log, n_log)
        num_vac_deaths2.append(D[-1])
    print("Finished 40% effectiveness")

    # ========== 60% effectiveness  =========
    k = 0.6          # vaccine effectiveness
    num_vac_deaths3 = []
    for i in release_dates:
        b_log = i
        initial_conditions = np.array((S0, Siv0, A0, I0, R0, D0))
        S, Siv, A, I, R, D = integrator(t, initial_conditions, Ba, Bi, p, mu, nu_i, nu_r, g, a, k, a_log, b_log, n_log)
        num_vac_deaths3.append(D[-1])
    print("Finished 60% effectiveness")

    # ========== 80% effectiveness  =========
    k = 0.8          # vaccine effectiveness
    num_vac_deaths4 = []
    for i in release_dates:
        b_log = i
        initial_conditions = np.array((S0, Siv0, A0, I0, R0, D0))
        S, Siv, A, I, R, D = integrator(t, initial_conditions, Ba, Bi, p, mu, nu_i, nu_r, g, a, k, a_log, b_log, n_log)
        num_vac_deaths4.append(D[-1])
    print("Finished 80% effectiveness")

    plt.plot(release_dates, num_vac_deaths4, label='Effectiveness = 80%', color='xkcd:royal blue')
    plt.plot(release_dates, num_vac_deaths3, label='Effectiveness = 60%', color='xkcd:bright blue')
    plt.plot(release_dates, num_vac_deaths2, label='Effectiveness = 40%', color='xkcd:sky blue')
    plt.plot(release_dates, num_vac_deaths1, label='Effectiveness = 20%', color='silver')

    plt.axvline(peak, label='Peak Of Infections', color='red', alpha=0.3)
    # plt.title("Vaccine Effectiveness and Total Deaths")
    plt.xlabel("Vaccine Release Date")
    plt.ylabel("Total Final Deaths")
    plt.legend()
    plt.xlim(0, 400)
    plt.show()

