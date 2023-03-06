import numpy as np
import matplotlib.pyplot as plt
import discrete_pendulum
import algorithms
import plotter


##########################
#       MODEL_FREE       #
##########################
env        = discrete_pendulum.Pendulum(n_theta=21, n_thetadot=21, n_tau=21)
s          = env.reset()
gamma      = 0.95
theta      = 1e-5
total_eps  = 5000
alpha      = 0.5
epsilon    = 1
Ql_sarsa   = []
Gl_sarsa   = []
Ql_qlearn  = []
Gl_qlearn  = []
eps_list   = [0, 0.2, 0.4, 0.6, 0.8, 1]
alpha_list = [0.2, 0.4, 0.5, 0.6, 0.8, 1]


##########################
#         SARSA          #
##########################

Q_sarsa, Ql_sarsa, Gl_sarsa = algorithms.SARSA(env, total_eps, epsilon, True, alpha, gamma, Ql_sarsa, Gl_sarsa)
P_star_sarsa      = np.argmax(Q_sarsa, axis = 1)


# TD(0) evaluation
V_sarsa           = algorithms.TD0(env, P_star_sarsa, alpha, gamma, total_eps)
title             = r"$\epsilon = \frac{1}{t}$, " + r"$\alpha = {}, \gamma = {}$".format(alpha, gamma)

# Learning curve plot (Average Qmax)
filename = "figures/pendulum/SARSA_learning_curve.png"
plotter.plot_q_vs_episodes(Ql_sarsa, "SARSA", "No. of episodes", "Average $Q_{max}$", "Pendulum, " + title, 'r', filename)

# Learning curve plot (Return)
filename = "figures/pendulum/SARSA_learning_curve_g.png"
plotter.plot_q_vs_episodes(Gl_sarsa, "SARSA", "No. of episodes", "Returns $G$", "Pendulum, " + title, 'r', filename)

# State value function plot
filename = "figures/pendulum/SARSA_state_value.png"
plotter.plot_q_vs_episodes(V_sarsa, "SARSA", "State, s", r"$V(s)$", r"Pendulum, $V^{*}(s)$, TD(0), " + title, 'r', filename)

# Policy plot
filename = "figures/pendulum/SARSA_policy.png"
plotter.plot_q_vs_episodes(P_star_sarsa, "SARSA", "State, s", "$\pi(s)$", r"Pendulum, $\pi^{*}$, " + title, 'r', filename)



# Plot policy and trajectory
plt.figure(figsize = (10, 8))
log = plotter.plot_trajectory(env, P_star_sarsa)
plt.plot(log['t'], log['s'], '-o')
plt.plot(log['t'][:-1], log['a'])
plt.plot(log['t'][:-1], log['r'])
plt.title("SARSA, Pendulum, "  + r"$\alpha = {}$, $\gamma = {}$".format(alpha, gamma), fontsize = 25)
plt.grid()
plt.legend(['s', 'a', 'r'])
plt.savefig('figures/pendulum/pendulum_sarsa_policy_trajec.png')

# Epsilon plots
val_name   = r"$\epsilon = $"
fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.set_figheight(15)
fig.set_figwidth(20)
fig.suptitle("SARSA, Pendulum, "  + r"$\alpha = {}$, $\gamma = {}$".format(alpha, gamma), fontsize = 25)
for eps in eps_list:
    Gl_s        = []
    Ql_s        = []
    Q_sar, Ql_s, Gl_s = algorithms.SARSA(env, total_eps, eps, False, alpha, gamma, Ql_s, Gl_s)
    plotter.plot_val_q_vs_episodes(Ql_s, eps, val_name, ax1)
    plotter.plot_val_q_vs_episodes(Gl_s, eps, val_name, ax2)
    
    
eps = np.arange(len(Ql_sarsa)) + 1
ax1.plot(eps, Ql_sarsa, label = r"$\epsilon = \frac{1}{t}$")
ax2.plot(eps, Gl_sarsa, label = r"$\epsilon = \frac{1}{t}$")

#ax1.set_xlabel("No. of episodes", fontsize = 15)
ax1.set_ylabel(r"Average $Q_{max}$", fontsize = 20)
ax1.grid(True)
ax1.legend()

ax2.set_xlabel("No. of episodes", fontsize = 20)
ax2.set_ylabel(r"Returns, $G$", fontsize = 20)
ax2.grid(True)
ax2.legend()

fig.tight_layout()
fig.savefig("figures/pendulum/pendulum_sarsa_epsilon.png", dpi=125)
plt.show()


# Alpha plots
val_name   = r"$\alpha = $"
fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.set_figheight(15)
fig.set_figwidth(20)
fig.suptitle("SARSA, Pendulum, "  + r"$\epsilon = \frac{1}{t}$, " + "$\gamma = {}$".format(gamma), fontsize = 25)
for alps in alpha_list:
    Ql_s        = []
    Gl_s        = []
    Q_sar, Ql_s, Gl_s = algorithms.SARSA(env, total_eps, epsilon, True, alps, gamma, Ql_s, Gl_s)
    plotter.plot_val_q_vs_episodes(Ql_s, alps, val_name, ax1)
    plotter.plot_val_q_vs_episodes(Gl_s, alps, val_name, ax2)

ax1.set_ylabel(r"Average $Q_{max}$", fontsize = 20)
ax1.grid(True)
ax1.legend()

ax2.set_xlabel("No. of episodes", fontsize = 20)
ax2.set_ylabel(r"Returns, $G$", fontsize = 20)
ax2.grid(True)
ax2.legend()

fig.tight_layout()
fig.savefig("figures/pendulum/pendulum_sarsa_alpha.png", dpi=125)
plt.show()



##########################
#       Q_LEARNING       #
##########################

Q_ql, Ql_qlearn, Gl_qlearn  = algorithms.Q_learning(env, total_eps, epsilon, True, alpha, gamma, Ql_qlearn, Gl_qlearn)
P_star_ql         = np.argmax(Q_ql, axis = 1)
V_qlearn          = algorithms.TD0(env, P_star_ql, alpha, gamma, total_eps)
title             = r"$\epsilon = \frac{1}{t}$, " + r"$\alpha = {}, \gamma = {}$".format(alpha, gamma)

#Learning curve plot
filepath = "figures/pendulum/ql_learning_curve.png"
plotter.plot_q_vs_episodes(Ql_qlearn, "Q-learning", "No. of episodes", "Average $Q_{max}$", "Pendulum, " + title, 'k', filepath)

#Learning curve g plot
filepath = "figures/pendulum/ql_learning_curve_g.png"
plotter.plot_q_vs_episodes(Gl_qlearn, "Q-learning", "No. of episodes", "Returns $G$", "Pendulum, " + title, 'k', filepath)

# State value function plot
filepath = "figures/pendulum/ql_state_value.png"
plotter.plot_q_vs_episodes(V_qlearn, "Q-learning", "State, s", r"$V(s)$", r"Pendulum, $V^{*}(s)$, TD(0), " + title, 'r', filepath)

# Policy plot
filepath = "figures/pendulum/ql_policy.png"
plotter.plot_q_vs_episodes(P_star_ql, "Q-learning", "State, s", "$\pi(s)$", r"Pendulum, $\pi^{*}$, " + title, 'r', filepath)

# Plot policy and trajectory
plt.figure(figsize = (10, 8))
log = plotter.plot_trajectory(env, P_star_ql)
plt.plot(log['t'], log['s'], '-o')
plt.plot(log['t'][:-1], log['a'])
plt.plot(log['t'][:-1], log['r'])
plt.title("Q-Learning, Pendulum, "  + r"$\alpha = {}$, $\gamma = {}$".format(alpha, gamma), fontsize = 20)
plt.grid()
plt.legend(['s', 'a', 'r'])
plt.savefig('figures/pendulum/pendulum_qlearning_policy_trajectory.png')

# Epsilon plots
val_name   = r"$\epsilon = $"
fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.set_figheight(15)
fig.set_figwidth(20)
fig.suptitle("Q-learning, Pendulum, "  + r"$\alpha = {}$, $\gamma = {}$".format(alpha, gamma), fontsize = 25)
for eps in eps_list:
    Ql_s        = []
    Gl_s        = []
    Q_sar, Ql_s, Gl_s = algorithms.Q_learning(env, total_eps, eps, False, alpha, gamma, Ql_s, Gl_s)
    plotter.plot_val_q_vs_episodes(Ql_s, eps, val_name, ax1)
    plotter.plot_val_q_vs_episodes(Gl_s, eps, val_name, ax2)
    
   
eps = np.arange(len(Ql_qlearn)) + 1
ax1.plot(eps, Ql_qlearn, label = r"$\epsilon = \frac{1}{t}$")
ax2.plot(eps, Gl_qlearn, label = r"$\epsilon = \frac{1}{t}$")

#ax1.set_xlabel("No. of episodes", fontsize = 15)
ax1.set_ylabel(r"Average $Q_{max}$", fontsize = 20)
ax1.grid(True)
ax1.legend()

ax2.set_xlabel("No. of episodes", fontsize = 20)
ax2.set_ylabel(r"Returns $G$", fontsize = 20)
ax2.grid(True)
ax2.legend()

fig.tight_layout()
fig.savefig("figures/pendulum/pendulum_qlearning_epsilon.png", dpi=125)
plt.show()


# Alpha plots
val_name   = r"$\alpha = $"
fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.set_figheight(15)
fig.set_figwidth(20)
fig.suptitle("Q-learning, Pendulum, "  + r"$\epsilon = \frac{1}{t}$, " + "$\gamma = {}$".format(gamma), fontsize = 25)
for alps in alpha_list:
    Ql_s        = []
    Gl_s        = []
    Q_sar, Ql_s, Gl_s = algorithms.Q_learning(env, total_eps, epsilon, True, alps, gamma, Ql_s, Gl_s)
    plotter.plot_val_q_vs_episodes(Ql_s, alps, val_name, ax1)
    plotter.plot_val_q_vs_episodes(Gl_s, alps, val_name, ax2)

#ax1.set_xlabel("No. of episodes", fontsize = 15)
ax1.set_ylabel(r"Average $Q_{max}$", fontsize = 20)
ax1.grid(True)
ax1.legend()

ax2.set_xlabel("No. of episodes", fontsize = 20)
ax2.set_ylabel(r"Returns $G$", fontsize = 20)
ax2.grid(True)
ax2.legend()

fig.tight_layout()
fig.savefig("figures/pendulum/pendulum_qlearning_alpha.png", dpi=125)
plt.show()


