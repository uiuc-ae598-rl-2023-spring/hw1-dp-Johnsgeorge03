import numpy as np
import matplotlib.pyplot as plt
import discrete_pendulum
import algorithms
import plotter


##########################
#       MODEL_FREE       #
##########################
env       = discrete_pendulum.Pendulum(n_theta=15, n_thetadot=21)
s         = env.reset()
gamma     = 0.95
theta     = 1e-5
total_eps = 3000
alpha     = 0.5
epsilon   = 1

Ql_sarsa  = []
Ql_qlearn = []

eps_list   = [0, 0.2, 0.4, 0.6, 0.8, 1]
alpha_list = [0.2, 0.4, 0.5, 0.6, 0.8, 1]


##########################
#         SARSA          #
##########################

Q_sarsa, Ql_sarsa = algorithms.SARSA(env, total_eps, epsilon, True, alpha, gamma, Ql_sarsa)
P_star_sarsa      = np.argmax(Q_sarsa, axis = 1)
#print("P_star_SARSA\n", P_star_sarsa.reshape(5,5))

# TD(0) evaluation
V_sarsa           = algorithms.TD0(env, P_star_sarsa, alpha, gamma, total_eps)
title             = r"$\epsilon = \frac{1}{t}$, " + r"$\alpha = {}, \gamma = {}$".format(alpha, gamma)

# Learning curve plot
plotter.plot_q_vs_episodes(Ql_sarsa, "SARSA", "No. of episodes", "Average $Q_{max}$", "Pendulum, " + title, 'r')
plt.savefig("figures/pendulum/SARSA_learning_curve.png", dpi=400)
# State value function plot
plotter.plot_q_vs_episodes(V_sarsa, "SARSA", "State, s", r"$V(s)$", r"Pendulum, $V^{*}(s)$, TD(0), " + title, 'r')
plt.savefig("figures/pendulum/SARSA_state_value.png", dpi=400)
# Policy plot
plotter.plot_q_vs_episodes(P_star_sarsa, "SARSA", "State, s", "$\pi(s)$", r"Pendulum, $\pi^{*}$, " + title, 'r')
plt.savefig("figures/pendulum/SARSA_policy.png", dpi=400)


# Plot policy and trajectory
plt.figure(figsize = (10, 8))
log = plotter.plot_trajectory(env, P_star_sarsa)
plt.plot(log['t'], log['s'], '-o')
plt.plot(log['t'][:-1], log['a'])
plt.plot(log['t'][:-1], log['r'])
plt.title("SARSA, Pendulum, "  + r"$\alpha = {}$, $\gamma = {}$".format(alpha, gamma), fontsize = 20)
plt.grid()
plt.legend(['s', 'a', 'r'])
plt.savefig('figures/pendulum/pendulum_sarsa.png')

# Epsilon plots
val_name   = r"$\epsilon = $"
plt.figure(figsize=(10, 8))
for eps in eps_list:
    Ql_s        = []
    Q_sar, Ql_s = algorithms.SARSA(env, total_eps, eps, False, alpha, gamma, Ql_s)
    plotter.plot_val_q_vs_episodes(Ql_s, eps, val_name)
    
eps = np.arange(len(Ql_sarsa)) + 1
plt.plot(eps, Ql_sarsa, label = r"$\epsilon = \frac{1}{t}$")
plt.xlabel("No. of episodes", fontsize = 15)
plt.ylabel(r"Average $Q_{max}$", fontsize = 15)
plt.title("SARSA, Pendulum, "  + r"$\alpha = {}$, $\gamma = {}$".format(alpha, gamma), fontsize = 20)
plt.grid()
plt.legend()
plt.savefig("figures/pendulum/pendulum_sarsa_epsilon.png", dpi=400)
plt.show()

# Alpha plots
val_name   = r"$\alpha = $"
plt.figure(figsize=(10, 8))
for alps in alpha_list:
    Ql_s        = []
    Q_sar, Ql_s = algorithms.SARSA(env, total_eps, epsilon, True, alps, gamma, Ql_s)
    plotter.plot_val_q_vs_episodes(Ql_s, alps, val_name)
    

plt.xlabel("No. of episodes", fontsize = 15)
plt.ylabel(r"Average $Q_{max}$", fontsize = 15)
plt.title("SARSA, Pendulum, "  + r"$\epsilon = \frac{1}{t}$, " + "$\gamma = {}$".format(gamma), fontsize = 20)
plt.grid()
plt.legend()
plt.savefig("figures/pendulum/pendulum_sarsa_alpha.png", dpi=400)
plt.show()



##########################
#       Q_LEARNING       #
##########################

Q_ql, Ql_qlearn   = algorithms.Q_learning(env, total_eps, epsilon, True, alpha, gamma, Ql_qlearn)
P_star_ql         = np.argmax(Q_ql, axis = 1)
#print("P_star_QL\n", P_star_ql.reshape(5,5))
V_qlearn          = algorithms.TD0(env, P_star_ql, alpha, gamma, total_eps)
title             = r"$\epsilon = \frac{1}{t}$, " + r"$\alpha = {}, \gamma = {}$".format(alpha, gamma)

#Learning curve plot
plotter.plot_q_vs_episodes(Ql_qlearn, "Q-learning", "No. of episodes", "Average $Q_{max}$", "Pendulum, " + title, 'k')
plt.savefig("figures/pendulum/ql_learning_curve.png", dpi=400)
# State value function plot
plotter.plot_q_vs_episodes(V_qlearn, "Q-learning", "State, s", r"$V(s)$", r"Pendulum, $V^{*}(s)$, TD(0), " + title, 'r')
plt.savefig("figures/pendulum/ql_state_value.png", dpi=400)
# Policy plot
plotter.plot_q_vs_episodes(P_star_ql, "Q-learning", "State, s", "$\pi(s)$", r"Pendulum, $\pi^{*}$, " + title, 'r')
plt.savefig("figures/pendulum/ql_policy.png", dpi=400)


# Plot policy and trajectory
plt.figure(figsize = (10, 8))
log = plotter.plot_trajectory(env, P_star_ql)
plt.plot(log['t'], log['s'], '-o')
plt.plot(log['t'][:-1], log['a'])
plt.plot(log['t'][:-1], log['r'])
plt.title("Q-Learning, Pendulum, "  + r"$\alpha = {}$, $\gamma = {}$".format(alpha, gamma), fontsize = 20)
plt.grid()
plt.legend(['s', 'a', 'r'])
plt.savefig('figures/pendulum/pendulum_qlearning.png')


# Epsilon plots
val_name   = r"$\epsilon = $"
plt.figure(figsize=(10, 8))
for eps in eps_list:
    Ql_s        = []
    Q_sar, Ql_s = algorithms.Q_learning(env, total_eps, eps, False, alpha, gamma, Ql_s)
    plotter.plot_val_q_vs_episodes(Ql_s, eps, val_name)
    
eps = np.arange(len(Ql_sarsa)) + 1
plt.plot(eps, Ql_qlearn, label = r"$\epsilon = \frac{1}{t}$")
plt.xlabel("No. of episodes", fontsize = 15)
plt.ylabel(r"Average $Q_{max}$", fontsize = 15)
plt.title("Q-learning, Pendulum, "  + r"$\alpha = {}$, $\gamma = {}$".format(alpha, gamma), fontsize = 20)
plt.grid()
plt.legend()
plt.savefig("figures/pendulum/pendulum_qlearning_epsilon.png", dpi=400)
plt.show()


# Alpha plots
val_name   = r"$\alpha = $"
plt.figure(figsize=(10, 8))
for alps in alpha_list:
    Ql_s        = []
    Q_sar, Ql_s = algorithms.Q_learning(env, total_eps, epsilon, True, alps, gamma, Ql_s)
    plotter.plot_val_q_vs_episodes(Ql_s, alps, val_name)
    

plt.xlabel("No. of episodes", fontsize = 15)
plt.ylabel(r"Average $Q_{max}$", fontsize = 15)
plt.title("Q-learning, Pendulum, "  + r"$\epsilon = \frac{1}{t}$, " + "$\gamma = {}$".format(gamma), fontsize = 20)
plt.grid()
plt.legend()
plt.savefig("figures/pendulum/pendulum_qlearning_alpha.png", dpi=400)
plt.show()


