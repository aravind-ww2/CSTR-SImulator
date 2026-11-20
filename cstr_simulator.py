import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def calculate_concentration(V, Q, CA0, k):
    return (Q * CA0) / (Q + k * V)


def calculate_conversion(CA0, CA):
    return (CA0 - CA) / CA0


def calculate_reaction_rate(k, CA):
    return k * CA


def calculate_residence_time(V, Q):
    return V / Q


def cstr_ode(t, y, V, Q, CA0, k):
    CA = y[0]
    dCA_dt = (Q / V) * (CA0 - CA) - k * CA
    return [dCA_dt]


def get_positive_float(prompt):
    while True:
        try:
            value = float(input(prompt))
            if value > 0:
                return value
            print("Value must be greater than zero.")
        except ValueError:
            print("Please enter a valid number.")


# inputs
V = get_positive_float("Enter the volume of the reactor in L: ")
Q = get_positive_float("Enter flow rate in L/min: ")
CA0 = get_positive_float("Enter inlet concentration in mol/L: ")
k = get_positive_float("Enter rate constant in 1/min: ")

# calculations
CA = calculate_concentration(V, Q, CA0, k)
X = calculate_conversion(CA0, CA)
r = calculate_reaction_rate(k, CA)
tau = calculate_residence_time(V, Q)
inlet_minus_outlet = Q * (CA0 - CA)
reaction_consumption = k * V * CA

# Varying Q
def simulate_flow_step(V, k, CA0, Q_old, Q_new, CA_initial, t_step, t_end):
    # old flow rate
    CA_initial = calculate_concentration(V, Q_old, CA0, k)
    t1 = np.linspace(0, t_step, 200)
    sol1 = solve_ivp(cstr_ode, (0, t_step), [CA_initial], t_eval=t1,
                     args=(V, Q_old, CA0, k))

    # new flow rate
    t2 = np.linspace(t_step, t_end, 200)
    sol2 = solve_ivp(cstr_ode, (t_step, t_end), [sol1.y[0][-1]], t_eval=t2,
                     args=(V, Q_new, CA0, k))

    t = np.concatenate([sol1.t, sol2.t[1:]])
    CA = np.concatenate([sol1.y[0], sol2.y[0][1:]])
    return t, CA


# Flow rate step experiment
Q_old = Q
Q_new = get_positive_float("Enter NEW flow rate after step (L/min): ")

tau_old = V / (Q + k * V)
tau_new = V / (Q_new + k * V)
t_step_ = 5 * tau_old
t_end_ = t_step_ + 5 * tau_new
t_Q, CA_Q = simulate_flow_step(V, k, CA0, Q_old, Q_new,
                               CA_initial=0.0,
                               t_step=t_step_, t_end=t_end_)

CA_ss_old = calculate_concentration(V, Q_old, CA0, k)
CA_ss_new = calculate_concentration(V, Q_new, CA0, k)
# Accumulation: Q is different before and after the step
Q_t = np.where(t_Q <= t_step_, Q_old, Q_new)
inflow = Q_t * CA0
outflow = Q_t * CA_Q
consumption = k * V * CA_Q
accumulation = inflow - outflow - consumption

print("\n===== CSTR SIMULATOR =====")
print(f"Reactor volume: {V} L")
print(f"Flow rate: {Q} L/min")
print(f"Inlet concentration: {CA0} mol/L")
print(f"Rate constant: {k} 1/min")

print("\n===== RESULTS =====")
print(f"Outlet concentration: {CA:.3f} mol/L")
print(f"Conversion: {X * 100:.2f}%")
print(f"Reaction rate: {r:.3f} mol/L/min")
print(f"Residence time: {tau:.2f} min")

print("\n===== MOLE BALANCE CHECK =====")
print(f"Flow-side balance: {inlet_minus_outlet:.6f} mol/min")
print(f"Reaction-side balance: {reaction_consumption:.6f} mol/min")
print(f"Difference: {abs(inlet_minus_outlet - reaction_consumption):.10f} mol/min")

time_constant = V / (Q + k * V)
t_end = 10 * time_constant
t_span = (0, t_end)
t_eval = np.linspace(0, t_end, 300)
solution = solve_ivp(
    cstr_ode,
    t_span,
    [0.0],
    t_eval=t_eval,
    args=(V, Q, CA0, k),
)

# Flow rate analysis
print("\n===== FLOW RATE ANALYSIS =====")
flow_rates = [1, 2, 2.5, 5, 10, 20]
conversions = []
for flow in flow_rates:
    CA_new = calculate_concentration(V, flow, CA0, k)
    X_new = calculate_conversion(CA0, CA_new) * 100
    conversions.append(X_new)
    print(f"Flow rate: {flow} L/min | Conversion: {X_new:.2f}%")

# Plot 1: conversion vs flow rate
plt.figure()
plt.plot(flow_rates, conversions, marker="o")
plt.xlabel("Flow rate (L/min)")
plt.ylabel("Conversion (%)")
plt.title("CSTR: Conversion vs Flow Rate")
plt.grid(True)
plt.tight_layout()
plt.savefig("conversion_vs_flow_rate.png", dpi=200)

# Plot 2: transient vs steady state
plt.figure()
plt.plot(solution.t, solution.y[0], label="ODE solution")
plt.axhline(CA, color="r", linestyle="--", label=f"Steady state ({CA:.3f} mol/L)")
plt.xlabel("Time (min)")
plt.ylabel("Concentration (mol/L)")
plt.title("CSTR: Concentration vs Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("concentration_vs_time.png", dpi=200)

# Plot 3: concentration response to flow-rate step
plt.figure()
plt.plot(t_Q, CA_Q, label="CA(t)")
plt.axhline(CA_ss_old, color="r", linestyle="--", label=f"Steady state 1 ({CA_ss_old:.3f})")
plt.axhline(CA_ss_new, color="g", linestyle="--", label=f"Steady state 2 ({CA_ss_new:.3f})")
plt.axvline(t_step_, color="gray", linestyle=":", label="Flow step")
plt.xlabel("Time (min)")
plt.ylabel("Concentration (mol/L)")
plt.title("CSTR: Response to Flow Rate Step")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("flow_step_concentration.png", dpi=200)

# Plot 4: accumulation
plt.figure()
plt.plot(t_Q, accumulation, color="tab:red", label="Accumulation")
plt.axhline(0, color="gray", linestyle=":")
plt.axvline(t_step_, color="gray", linestyle=":", label="Flow step")
plt.xlabel("Time (min)")
plt.ylabel("Accumulation (mol/min)")
plt.title("CSTR: Accumulation During Flow Step")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("flow_step_accumulation.png", dpi=200)

plt.show()
