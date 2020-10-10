import numpy as np
import matplotlib.pyplot as plt  # plotting
from scipy.integrate import odeint  # ODE integrator
import openpyxl  # module to read excel files
import matplotlib.dates as dt  # module for dealing with dates
from scipy.optimize import curve_fit
import matplotlib.dates as mdates

# Importing/Plotting data ==============================================================================================
workbook = openpyxl.load_workbook(filename="covid.xlsx")  # loads in an excel file
sheet = workbook['Sheet5']  # assigns the first sheet to an object
dates = []  # save the data
S_data = []
I_data = []
R_data = []
D_data = []

case_data = []  # cases
dead_data = []  # dead

for col in sheet.iter_cols(min_col=2, values_only=True):  # loop through columns of spreadsheet
    dates.append(dt.datestr2num(str(col[0])))
    case_data.append(col[1])
    dead_data.append(col[2])

date_ints = np.linspace(0, len(dates) - 1, len(dates))  # corresponding integers to datenum objects

# plot data
fig, ax = plt.subplots()
fig.autofmt_xdate()  # rotate date labels so they are diagonal

# plt.plot_date(dates, S_data, label='Total Susceptible')
# plt.plot_date(dates, I_data, label='Total Infected')
# plt.plot_date(dates, R_data, label='Total Recovered')
# plt.plot_date(dates, D_data, label='Total Dead')

plt.plot_date(dates, case_data, label='Total Cases')
plt.plot_date(dates, dead_data, label='Total Dead')

plt.xlabel('Date')
plt.ylabel('Total Number People')
# plt.title("U.S. Covid19 Outbreak *NYT Data")
plt.ylim(0, 4000000)  # scale to look at infected/recovered
plt.legend()
plt.show()


# Initialization and defining functions ================================================================================
def sir(init_cond, t, a, Ba, Bi, g, p, m, n_i, n_r):
    """
    :param Ba: (beta) infectivity rate
    :param m: percentage of asymptomatic people that get sick
    :param n_i: how long people are asymptomatic for before getting symptoms (infected)
    :param n_r: how long people are asymptomatic before recovering if they never show symptoms
    :param a: (alpha) death rate (% chance of death)
    :param g: (gamma) proportion of infected people RECOVERING per day
    :param p: (rho) how many days it takes to die

    """

    S, A, I, R, D = init_cond

    dS = - Bi*I*(S/N) - Ba*A*(S/N)

    dA = + Bi*I*(S/N) + Ba*A*(S/N) - m*A*n_i - (1-m)*A*n_r

    dI = + m*A*n_i - a*p*I - (1-a)*g*I

    dR = + (1-m)*A*n_r + (1-a)*g*I

    dD = + a*p*I

    return np.array((dS, dA, dI, dR, dD))


def sir_trajectory(t, a, Ba, Bi, g, p, m, n_i, n_r):
    sir_pop = odeint(sir, initial_conditions, t, args=(a, Ba, Bi, g, p, m, n_i, n_r))
    S = sir_pop[:, 0]
    A = sir_pop[:, 1]
    I = sir_pop[:, 2]
    R = sir_pop[:, 3]
    D = sir_pop[:, 4]

    return S, A, I, R, D


def exponential(x, a, b, c):
    return a + b * np.exp(c * x)


def s_trajectory(t, a, Ba, Bi, g, p, m, n_i, n_r):  # change the inputs here depending on what we want to fit
    return sir_trajectory(t, a, Ba, Bi, g, p, m, n_i, n_r)[0]


def i_trajectory(t, a, Ba, Bi, g, p, mu, nu_i, nu_r):
    return sir_trajectory(t, a, Ba, Bi, g, p, mu, nu_i, nu_r)[2]


# GLOBAL VARIABLES =====================================================================================================
# Time step
SimulationTime = 400.0  # test simulation time
dt = 1
t = np.arange(0.0, SimulationTime + dt, dt)

# initial conditions
N = 3.28196e08  # population of US
I0 = 1000       # initial infected --> note 'infected' means infected/showing symptoms, asymptomatic are also infected
A0 = 0          # initial asymptomatic
R0 = 0          # initial recovered
S0 = N - I0     # initial susceptible
D0 = 0          # initial dead
initial_conditions = np.array((S0, A0, I0, R0, D0))

mu = 0.85       # percentage of asymptomatic people that get sick
nu_i = 0.1961   # proportion of people transitioning from asymptomatic to infected
nu_r = 0.1111   # proportion of people transitioning from asymptomatic to recovered
p = 0.0562      # proportion of people dying per day. aka. it takes 7 days to die
g = 0.0870      # proportion of people recovering per day. aka. it takes 5 days to recover
a = 0.0064      # death rate ie. 10% of infected people will die
r0 = 2          #
B = r0/(1/nu_i + 1/g)   # = 0.1205
Ba = B                  # infectivity of asymptomatic people
Bi = B                  # infectivity of infected (symptomatic) people
# note: using B as a heuristic for Ba and Bi, data unavailable
print("B: " + str(B))


# SIR MODEL ============================================================================================================
S, A, I, R, D = sir_trajectory(t, a, Ba, Bi, g, p, mu, nu_i, nu_r)

plt.plot(t, S, label='Susceptible', color='b')
plt.plot(t, A, label='Asymptomatic', color='c')
plt.plot(t, I, label='Infected', color='m')
plt.plot(t, R, label='Recovered', color='g')
plt.plot(t, D, label='Dead', color='r')
plt.xlabel('Time (days)')
plt.ylabel('Number of People')
# plt.title("Base SAIRD Model")
plt.legend()
plt.show()

print("Final Deaths: " + str(D[400]))


# EXPONENTIAL FIT ======================================================================================================
a_guess = 0
b_guess = 300
c_guess = 0.01
a_fit, b_fit, c_fit = curve_fit(exponential, date_ints, I_data, p0=(a_guess, b_guess, c_guess), bounds=[0, np.inf])[0]

date_ints2 = np.arange(1, 111, 1)
zeros = np.zeros(110 - len(I_data))
I_data2 = np.concatenate((I_data, zeros), axis=None)

fig, ax = plt.subplots()
fig.autofmt_xdate()  # rotate date labels so they are diagonal
plt.scatter(date_ints2, I_data2, label='data', c='orange')
plt.plot(date_ints2, exponential(date_ints2, a_fit, b_fit, c_fit), label='Exponential Fit')
plt.xlabel('Time (days)')
plt.ylabel('Total Number People')
plt.title("Infected Data: Exponential Fit")
plt.legend()
plt.show()

# SIR MODEL FIT ========================================================================================================

fit_guess = [a, Ba, Bi, g, p, mu, nu_i, nu_r]
# fit_guess = [Ba, Bi]

fit_output = curve_fit(i_trajectory, date_ints, I_data, p0=fit_guess, bounds=[0, np.inf])[0]
a_fit = fit_output[0]
Ba_fit = fit_output[1]
Bi_fit = fit_output[2]
g_fit = fit_output[3]
p_fit = fit_output[4]
mu_fit = fit_output[5]
nu_i_fit = fit_output[6]
nu_r_fit = fit_output[7]

# Ba_fit = fit_output[0]
# Bi_fit = fit_output[1]

print("Fitted Data ________________________")
print("a fit: " + str(a_fit))
print("Ba fit: " + str(Ba_fit))
print("Bi fit: " + str(Bi_fit))
print("g fit: " + str(g_fit))
print("p fit: " + str(p_fit))
print("mu fit: " + str(mu_fit))
print("nu_i fit: " + str(nu_i_fit))
print("nu_r fit: " + str(nu_r_fit))

S, A, I, R, D = sir_trajectory(date_ints2, a_fit, Ba_fit, Bi_fit, g_fit, p_fit, mu_fit, nu_i_fit, nu_r_fit)

fig, ax = plt.subplots()
fig.autofmt_xdate()  # rotate date labels so they are diagonal
plt.scatter(date_ints2, I_data2, label='Infected data', c="orange")
plt.plot(date_ints2, I, label='SIR Fit')
plt.xlabel('Time (days)')
plt.ylabel('Total Number People')
plt.title("Infected Data: Base SAIRD Model Fit")
plt.legend()
# plt.ylim(0, 100000)  # scale to look at infected/recovered
plt.show()
