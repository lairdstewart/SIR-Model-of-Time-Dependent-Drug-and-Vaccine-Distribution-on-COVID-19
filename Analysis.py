import DrugModel as CD
import VaccineModel as CV
import numpy as np
import scipy.integrate as odeint
import matplotlib.pyplot as plt

# ======================================================================================================================
# VARIABLES
SimulationTime = 500.0
dt = 1
t = np.arange(0.0, SimulationTime + dt, dt)

# SAIRD Model Variables
Ba = 0.120527  # infectivity of asymptomatic people
Bi = 0.120527  # infectivity of infected (symptomatic) people
mu = 0.85  # percentage of asymptomatic people that get sick
nu_i = 0.1961  # proportion of people transitioning from asymptomatic to infected
nu_r = 0.1111  # proportion of people transitioning from asymptomatic to recovered
p = 0.0562  # proportion of people dying per day (no drug). aka. it takes 7 days to die
g = 0.0870  # proportion of people recovering per day. aka. it takes 11 days to recover
a = 0.0064  # death rate ie. 10% of infected people will die

# Vaccine Variables
k = 0.91  # effectiveness of the vaccine
a_log = 1  # a in logistic distribution equation (see function)
b_log = 250  # b in logistic distribution equation (see function)
n_log = 100000000  # total number of vaccines distributed

# Drug Variables
g_drug = 0.102  # gamma with the drug
a_drug = 0.0032  # alpha with the drug
p2 = p  # proportion of people dying per day (with drug) ie. slower
steep = 0.1  # 'steepness' of the logistic curve
infl = 250  # 'midpoint' ie. time of steepest part of the logistic curve

# Starting Conditions
N = 3.28196e08  # total population of the US
I0 = 1000  # infected --> note 'infected' means infected/showing symptoms, asymptomatic are also infected
R0 = 0  # Initial Recovered Population
S0 = N - I0  # Initial Susceptible Population (N - 1)
D0 = 0  # Initial Dead Population
Siv0 = 0  # Initial Ineffectively Vaccinated Population
A0 = 0  # Initial Asymptomatic Population
d_initial_conditions = np.array((S0, A0, I0, R0, D0))
v_initial_conditions = np.array((S0, Siv0, A0, I0, R0, D0))

# ======================================================================================================================

S, A, I, R, D = CD.integrator(t, d_initial_conditions, a, Ba, Bi, g, p, p2, a_drug, g_drug, mu, nu_i, nu_r, k, infl)
peak = np.argmax(I)

# S, Siv, A, I, R, D = CV.integrator(t, v_initial_conditions, Ba, Bi, p, mu, nu_i, nu_r, g, a, k, a_log, b_log, n_log)


# ======================================================================================================================
# Comparing a 'perfect' drug vs. 'perfect' vaccine
# perfect drug: drug 1/100 day recovery (fastest can go without overflow)
# perfect vaccine: effectivity = 100%, n_log = N (everyone gets it)

release_dates = np.arange(20, 401, 1)
a_drug = 0  # drug death rate
g_drug = 100  # drug 1 day recovery
n_log = N  # vaccine is distributed to everyone in population
k = 1  # vaccine is 100% effective

# ========== Perfect Vaccine  =========
perf_vacc_deaths = []
for i in release_dates:
    b_log = i
    S, Siv, A, I, R, D = CV.integrator(t, v_initial_conditions, Ba, Bi, p, mu, nu_i, nu_r, g, a, k, a_log, b_log, n_log)
    perf_vacc_deaths.append(D[-1])
print("Finished perfect vaccine")

# ========== Perfect Drug  =========
release_dates = np.arange(20, 401, 1)
perf_drug_deaths = []
for i in release_dates:
    infl = i
    S, A, I, R, D = CD.integrator(t, d_initial_conditions, a, Ba, Bi, g, p, p2, a_drug, g_drug, mu, nu_i, nu_r, k,
                                  infl)
    perf_drug_deaths.append(D[-1])
print("Finished perfect drug")

plt.plot(release_dates, perf_vacc_deaths, label='Perfect Vaccine', color='xkcd:royal blue')
plt.plot(release_dates, perf_drug_deaths, label='Perfect Drug', color='orange')
plt.axvline(peak, label='Peak Of Infections', color='red', alpha=0.3)
# plt.title("Perfect Drug vs. Perfect Vaccine")
plt.xlabel("Drug/Vaccine Release Date")
plt.ylabel("Total Final Deaths")
plt.legend()
# plt.xlim(0, 150)
plt.show()

# ======================================================================================================================
# 'Reasonable' Drug vs 'Reasonable' Vaccine (using the numbers we have found from real life)
release_dates = np.arange(20, 501, 1)
a_drug = 0.0032  # drug death rate
g_drug = 0.102  # drug recovery rate (9.8 days)
k = 0.91  # effectiveness

# ========== Reasonable Vaccine 50% N   =========
n_log = 0.5 * N
reas_vacc_deaths = []
for i in release_dates:
    b_log = i
    S, Siv, A, I, R, D = CV.integrator(t, v_initial_conditions, Ba, Bi, p, mu, nu_i, nu_r, g, a, k, a_log, b_log, n_log)
    reas_vacc_deaths.append(D[-1])
print("Finished reasonable vaccine 50% N ")

# ========== Reasonable Vaccine 70% N   =========
n_log = 0.7 * N
reas_vacc_deaths2 = []
for i in release_dates:
    b_log = i
    S, Siv, A, I, R, D = CV.integrator(t, v_initial_conditions, Ba, Bi, p, mu, nu_i, nu_r, g, a, k, a_log, b_log, n_log)
    reas_vacc_deaths2.append(D[-1])
print("Finished reasonable vaccine 70% N ")

# ========== Reasonable Vaccine 90% N   =========
n_log = 0.9 * N
reas_vacc_deaths3 = []
for i in release_dates:
    b_log = i
    S, Siv, A, I, R, D = CV.integrator(t, v_initial_conditions, Ba, Bi, p, mu, nu_i, nu_r, g, a, k, a_log, b_log, n_log)
    reas_vacc_deaths3.append(D[-1])
print("Finished reasonable vaccine 90% N ")

# ========== Reasonable Drug  =========
release_dates = np.arange(20, 501, 1)
reas_drug_deaths = []
for i in release_dates:
    infl = i
    S, A, I, R, D = CD.integrator(t, d_initial_conditions, a, Ba, Bi, g, p, p2, a_drug, g_drug, mu, nu_i, nu_r, k,
                                  infl)
    reas_drug_deaths.append(D[-1])
print("Finished reasonable drug")

plt.plot(release_dates, reas_vacc_deaths, label='50% vaccination capacity', color='xkcd:sky blue')
plt.plot(release_dates, reas_vacc_deaths2, label='70% vaccination capacity', color='xkcd:bright blue')
plt.plot(release_dates, reas_vacc_deaths3, label='90% vaccination capacity', color='xkcd:royal blue')
plt.plot(release_dates, reas_drug_deaths, label='Reasonable Drug', color='orange')
plt.axvline(peak, label='Peak Of Infections', color='red', alpha=0.3)
# plt.title("Reasonable Drug vs. Reasonable Vaccine")
plt.xlabel("Drug/Vaccine Release Date")
plt.ylabel("Total Final Deaths")
plt.legend()
plt.show()

# =============== Plotting differences in % effectiveness ======================
# Final # of Deaths: 956,222
drug = np.array(reas_drug_deaths)
vacc = np.array(reas_vacc_deaths)
vacc2 = np.array(reas_vacc_deaths2)
vacc3 = np.array(reas_vacc_deaths3)

eff_diff = (1-(drug / 956222)) - (1-(vacc / 956221))
eff_diff2 = (1-(drug / 956222)) - (1-(vacc2 / 956221))
eff_diff3 = (1-(drug / 956222)) - (1-(vacc3 / 956221))

plt.plot(release_dates, eff_diff, label='50% vaccination capacity', color='xkcd:sky blue')
plt.plot(release_dates, eff_diff2, label='70% vaccination capacity', color='xkcd:bright blue')
plt.plot(release_dates, eff_diff3, label='90% vaccination capacity', color='xkcd:royal blue')

plt.axvline(peak, label='Peak Of Infections', color='red', alpha=0.3)
plt.axhline(linestyle='--', y=0, xmin=0, xmax=500, color='gray', alpha=0.3)

# plt.title("Difference in % of Lives Saved for Drug vs. 50%, 70%, 90% Effective Vaccines")
plt.xlabel("Drug/Vaccine Release Date")
plt.ylabel("Difference in % Lives Saved")
plt.legend()
plt.show()

print("peak: " + str(peak))  # 193

print("at peak 50: " + str(eff_diff[peak-20]))  # - 20, because each list starts at time 20
print("at peak 70: " + str(eff_diff2[peak-20]))
print("at peak 900: " + str(eff_diff3[peak-20]))

print("50% effectiveness" + str(np.amax(eff_diff)))
print("70% effectiveness" + str(np.amax(eff_diff2)))
print("90% effectiveness" + str(np.amax(eff_diff3)))

print("50% effectiveness (min)" + str(np.amin(eff_diff)))
print("70% effectiveness (min)" + str(np.amin(eff_diff2)))
print("90% effectiveness (min)" + str(np.amin(eff_diff3)))

print("final deaths: " + str(reas_vacc_deaths[-1]))  # 856221

