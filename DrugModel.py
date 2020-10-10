import numpy as np
import matplotlib.pyplot as plt  # plotting
from scipy.integrate import odeint  # ODE integrator
import openpyxl  # module to read excel files
import matplotlib.dates as dt  # module for dealing with dates
from scipy.optimize import curve_fit

# ======================================================================================================================
def drug_sir(init_cond, t, a1, Ba, Bi, g1, p1, p2, a2, g2, m, n_i, n_r, k, infl):
    """
    :param m: percentage of asymptomatic people that get sick
    :param n_i: how long people are asymptomatic for before getting symptoms (infected)
    :param n_r: how long people are asymptomatic before recovering if they never show symptoms
    :param init_cond: 4 part list to keep track of current values
    :param t: needed for integrator (not used in function itself)
    :param a1: (alpha) death rate (% chance of death)
    :param a2: (alpha 2) death rate after the drug has been developed (% chance of death)
    :param b: (beta) infectivity rate
    :param g1: (gamma) proportion of infected people RECOVERING per day
    :param g2: (gamma 2) after the drug has been developed
    :param p1: (rho) how many days it takes to die
    :param p2: (rho) how many days with the drug it takes to die
    """
    drug = logistic_drug_availability(t, k, infl)  # percentage of people that can get the drug at a given time

    S, A, I, R, D = init_cond

    dS = - Bi*I*(S/N) - Ba*A*(S/N)

    dA = + Bi*I*(S/N) + Ba*A*(S/N) - m * A * n_i - (1-m) * A * n_r

    dI = + m * A * n_i - drug * a2 * p1 * I \
         - drug * (1 - a2) * g2 * I \
         - (1 - drug) * a1 * p2 * I \
         - (1 - drug) * (1 - a1) * g1 * I

    dR = + (1-m) * A * n_r \
         + drug * (1 - a2) * g2 * I \
         + (1 - drug) * (1 - a1) * g1 * I

    dD = + drug * a2 * p1 * I \
         + (1 - drug) * a1 * p2 * I

    return np.array((dS, dA, dI, dR, dD))


def linear_drug_availability(t):
    # given a time, return what percentage of the population could receive the drug
    if t < 100:
        return t / 100
    else:
        return 1  # percentage of population that will receive the drug after time = 100 is 1


def logistic_drug_availability(t, k, infl):
    return 1/(1+np.exp(k*(-t+infl)))


def integrator(t, init_cond, alpha, beta_a, beta_i, gamma, rho1, rho2, alpha_prime, gamma_prime, mu, nu_i, nu_r, k, infl):
    output = odeint(drug_sir, init_cond, t, args=(alpha, beta_a, beta_i, gamma, rho1, rho2, alpha_prime, gamma_prime, mu, nu_i, nu_r, k, infl))
    s = output[:, 0]
    a = output[:, 1]
    i = output[:, 2]
    r = output[:, 3]
    d = output[:, 4]
    return np.array((s, a, i, r, d))


# ======================================================================================================================
# VARIABLES
SimulationTime = 400  # simulation length
dt = 1                # time step
t = np.arange(0.0, SimulationTime + dt, dt)

# GLOBAL VARIABLES
Ba = 0.120527   # infectivity of asymptomatic people
Bi = 0.120527   # infectivity of infected (symptomatic) people
mu = 0.85       # percentage of asymptomatic people that get sick
nu_i = 0.1961   # proportion of people transitioning from asymptomatic to infected
nu_r = 0.1111   # proportion of people transitioning from asymptomatic to recovered
p1 = 0.0562     # proportion of people dying per day (no drug). aka. it takes 7 days to die
g = 0.0870      # proportion of people recovering per day. aka. it takes 11 days to recover
a = 0.0064      # death rate ie. 10% of infected people will die

# Drug Variables:
g_drug = 0.102      # gamma with the drug
a_drug = 0.0032     # alpha with the drug
p2 = p1             # proportion of people dying per day (with drug) ie. slower (heuristic)
k = 0.1             # 'steepness' of the logistic curve
infl = 250          # 'midpoint' ie. time of steepest part of the logistic curve

# Initial Conditions
N = 3.28196e08  # us population
I0 = 1000       # initial infected --> note 'infected' means infected/showing symptoms, asymptomatic are also infected
R0 = 0          # Initial Recovered
S0 = N - I0     # Initial Susceptible (Total - infected)
D0 = 0          # Initial Dead
A0 = 0          # Initial Asymptomatic

# ======================================================================================================================
initial_conditions = np.array((S0, A0, I0, R0, D0))
S, A, I, R, D = integrator(t, initial_conditions, a, Ba, Bi, g, p1, p2, a_drug, g_drug, mu, nu_i, nu_r, k, infl)

# ======================================================================================================================
# BASIC PLOT
if __name__ == '__main__':
    print("\nFinal Conditions at t = " + str(SimulationTime) + ": ")
    print("\tSusceptible = " + str(int(S[-1])))
    print("\tAsymptomatic = " + str(int(A[-1])))
    print("\tInfected = " + str(int(I[-1])))
    print("\tRecovered = " + str(int(R[-1])))
    print("\tDead = " + str(int(D[-1])))

    plt.plot(t, S, label='Susceptible', color='b')
    plt.plot(t, A, label='Asymptomatic', color='c')
    plt.plot(t, I, label='Infected', color='m')
    plt.plot(t, R, label='Recovered', color='g')
    plt.plot(t, D, label='Dead', color='r')
    # plt.plot(t, N * np.ones(len(t)), label='Total Population', color='c')
    plt.xlabel('Days')
    plt.ylabel('Number of People')
    # plt.title("SIR Model Test")
    plt.legend(loc='upper center')
    plt.legend()
    plt.show()

    peak = np.argmax(I)

# ======================================================================================================================
# DRUG AVAILABILITY PLOT
if __name__ == '__main__':
    plt.plot(t, logistic_drug_availability(t, 0.1, 200), label='a = 0.1')
    plt.ylim(0, 1)
    # plt.title("Drug Distribution")
    plt.xlabel("Time (days)")
    plt.ylabel("Percentage of patients treated with drug")
    plt.ylim(-0.25, 1.25)
    plt.grid()
    plt.legend()
    plt.show()


# ======================================================================================================================
# DEATH VS. DRUG RELEASE DATE (changing recovery rate gamma)
# below, we keep alpha2 (death rate with the drug) constant to isolate changes in gamma2
# for each new gamma, we run the simulation for the drug's release date on every day of the pandemic.
# If the drug is released on the last day (after the pandemic was over) we would expect no change


if __name__ == '__main__':
    release_dates = np.arange(0, 401, 1)
    # Constants:
    a_drug = 0.0064
    p2 = 0.0562
    k = 1

    # ========== gamma 1/11.5 (no change) ==========
    gamma_deaths1 = []
    g_drug = 0.0870     # same as g1
    for i in release_dates:
        r_d = i     # 'release date' of drug ie. inflection point
        initial_conditions = np.array((S0, A0, I0, R0, D0))
        S, A, I, R, D = integrator(t, initial_conditions, a, Ba, Bi, g, p1, p2, a_drug, g_drug, mu, nu_i, nu_r, k, r_d)
        gamma_deaths1.append(D[-1])
    print("Finished gamma = 1/11.5 (no change)")

    # ========== gamma 1/9 ==========
    gamma_deaths2 = []
    g_drug = 1/9
    k = 1  # 'steepness' of the logistic curve
    for i in release_dates:
        r_d = i     # 'release date' of drug ie. inflection point
        initial_conditions = np.array((S0, A0, I0, R0, D0))
        S, A, I, R, D = integrator(t, initial_conditions, a, Ba, Bi, g, p1, p2, a_drug, g_drug, mu, nu_i, nu_r, k, r_d)
        gamma_deaths2.append(D[-1])
    print("Finished gamma = 1/15")

    # ========== gamma 1/7  ==========
    gamma_deaths3 = []
    g_drug = 1/7        # NEW
    for i in release_dates:
        r_d = i     # 'release date' of drug ie. inflection point
        initial_conditions = np.array((S0, A0, I0, R0, D0))
        S, A, I, R, D = integrator(t, initial_conditions, a, Ba, Bi, g, p1, p2, a_drug, g_drug, mu, nu_i, nu_r, k, r_d)
        gamma_deaths3.append(D[-1])
    print("Finished gamma = 1/9")

    # ========== gamma 1/5  ==========
    gamma_deaths4 = []
    g_drug = 1/5   # NEW
    for i in release_dates:
        r_d = i     # 'release date' of drug ie. inflection point
        initial_conditions = np.array((S0, A0, I0, R0, D0))
        S, A, I, R, D = integrator(t, initial_conditions, a, Ba, Bi, g, p1, p2, a_drug, g_drug, mu, nu_i, nu_r, k, r_d)
        gamma_deaths4.append(D[-1])
    print("Finished gamma = 1/10")

    plt.plot(release_dates, gamma_deaths1, label='γ2 = γ1 = 1/11.5', color='silver')
    plt.plot(release_dates, gamma_deaths2, label='γ2 = 1/9', color='xkcd:sky blue')
    plt.plot(release_dates, gamma_deaths3, label='γ2 = 1/7', color='xkcd:bright blue')
    plt.plot(release_dates, gamma_deaths4, label='γ2 = 1/5', color='xkcd:royal blue')

    plt.axvline(peak, label='Peak Of Infections', color='red', alpha=0.3)
    # plt.title("Variable γ2 and Total Deaths")
    plt.xlabel("Drug Release Date")
    plt.ylabel("Total Final Deaths")
    plt.legend(loc='upper center')
    plt.legend()
    plt.show()

# ======================================================================================================================
# DEATH VS. DRUG RELEASE DATE (ONLY CHANGING ALPHA)
if __name__ == '__main__':
    release_dates = np.arange(0, 401, 1)
    # Constants:
    g_drug = 0.0870
    p2 = 0.0562
    k = 1

    # ========== A 0.0064 (no change) ==========
    deaths_base = []
    a_drug = 0.0064    # same as a (base)
    for i in release_dates:
        r_d = i     # 'release date' of drug ie. inflection point
        initial_conditions = np.array((S0, A0, I0, R0, D0))
        S, A, I, R, D = integrator(t, initial_conditions, a, Ba, Bi, g, p1, p2, a_drug, g_drug, mu, nu_i, nu_r, k, r_d)
        deaths_base.append(D[-1])
    print("Finished a = 0.0064 (no change)")

    # ========== DEATH 0.005 ==========
    deaths_1 = []
    a_drug = 0.005    # NEW
    for i in release_dates:
        r_d = i     # 'release date' of drug ie. inflection point
        initial_conditions = np.array((S0, A0, I0, R0, D0))
        S, A, I, R, D = integrator(t, initial_conditions, a, Ba, Bi, g, p1, p2, a_drug, g_drug, mu, nu_i, nu_r, k, r_d)
        deaths_1.append(D[-1])
    print("Finished a = 0.005")

    # ========== DEATH 0.004 ==========
    deaths_2 = []
    a_drug = 0.004    # NEW
    for i in release_dates:
        r_d = i     # 'release date' of drug ie. inflection point
        initial_conditions = np.array((S0, A0, I0, R0, D0))
        S, A, I, R, D = integrator(t, initial_conditions, a, Ba, Bi, g, p1, p2, a_drug, g_drug, mu, nu_i, nu_r, k, r_d)
        deaths_2.append(D[-1])
    print("Finished a = 0.004")

    # ========== DEATH 0.003 ==========
    deaths_3 = []
    a_drug = 0.003    # NEW
    for i in release_dates:
        r_d = i     # 'release date' of drug ie. inflection point
        initial_conditions = np.array((S0, A0, I0, R0, D0))
        S, A, I, R, D = integrator(t, initial_conditions, a, Ba, Bi, g, p1, p2, a_drug, g_drug, mu, nu_i, nu_r, k, r_d)
        deaths_3.append(D[-1])
    print("Finished a = 0.003")

    plt.plot(release_dates, deaths_base, label='α2 = α1 = 0.0064', color='silver')
    plt.plot(release_dates, deaths_1, label='α2 = 0.005', color='xkcd:sky blue')
    plt.plot(release_dates, deaths_2, label='α2 = 0.004', color='xkcd:bright blue')
    plt.plot(release_dates, deaths_3, label='α2 = 0.003', color='xkcd:royal blue')

    plt.axvline(peak, label='Peak Of Infections', color='red', alpha=0.3)
    # plt.title("Variable α2 and Total Deaths")
    plt.xlabel("Drug Release Date")
    plt.ylabel("Total Final Deaths")
    plt.legend(loc='upper center')
    plt.legend()
    plt.show()

    print("Min a2 = 0.005: " + str(min(deaths_3)))
    print("Min a2 = 0.01: " + str(min(deaths_2)))
    print("Min a2 = 0.015: " + str(min(deaths_1)))
    print("Min a2 = 0.0021: " + str(min(deaths_base)))

# ======================================================================================================================


