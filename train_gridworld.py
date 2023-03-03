import numpy as np
import matplotlib.pyplot as plt
import gridworld
import algorithms
import plotter
import sys
    
    
# Create environment
env      = gridworld.GridWorld(hard_version=False)
s        = env.reset()
gamma    = 0.95
tol      = 1e-8
max_iter = 4000
title    = r"$\gamma = {}$".format(gamma)



##########################
#      MODEL_BASED       #
##########################



##########################
#    Policy Iteration    #
##########################
Vl_pi    = []
V_star_pi, P_star_pi = algorithms.policy_iteration(env, gamma, tol, max_iter, Vl_pi)

# Learning curve plot
plotter.plot_q_vs_episodes(Vl_pi, "Policy_iteration", "No. of episodes", \
                           "Average $V(s)$", "Gridworld, Policy Iteration, " + title, 'r')



# Plot policy and trajectory
plt.figure(figsize = (10, 8))
log = plotter.plot_trajectory(env, P_star_pi)
plt.plot(log['t'], log['s'], '-o')
plt.plot(log['t'][:-1], log['a'])
plt.plot(log['t'][:-1], log['r'])
plt.title("Policy Iteration, Gridworld, "  + r"$\gamma = {}$".format(gamma), fontsize = 20)
plt.grid()
plt.legend(['s', 'a', 'r'])
plt.savefig('figures/gridworld/gridworld_pi_policy_trajec.png')


##########################
#    Value Iteration     #
##########################
Vl_vi  = []
V_star_vi, P_star_vi = algorithms.value_iteration(env, gamma, tol, max_iter, Vl_vi)

# Learning curve plot
plotter.plot_q_vs_episodes(Vl_vi, "Value_iteration", "No. of episodes", "Average $V(s)$", \
                           "Gridworld, Value Iteration, " + title, 'r')

# Plot policy and trajectory
plt.figure(figsize = (10, 8))
log = plotter.plot_trajectory(env, P_star_pi)
plt.plot(log['t'], log['s'], '-o')
plt.plot(log['t'][:-1], log['a'])
plt.plot(log['t'][:-1], log['r'])
plt.title("Value Iteration, Gridworld, "  + r"$\gamma = {}$".format(gamma), fontsize = 20)
plt.grid()
plt.legend(['s', 'a', 'r'])
plt.savefig('figures/gridworld/gridworld_vi_policy_trajec.png')

print("P_star_pi\n", P_star_pi.reshape(5,5))
print("P_star_vi\n", P_star_vi.reshape(5,5))

#input("Press Enter to continue...")

##########################
#       MODEL_FREE       #
##########################
total_eps  = 2000
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
print("P_star_SARSA\n", P_star_sarsa.reshape(5,5))

# TD(0) evaluation
V_sarsa           = algorithms.TD0(env, P_star_sarsa, alpha, gamma, total_eps)
title             = r"$\epsilon = \frac{1}{t}$, " + r"$\alpha = {}, \gamma = {}$".format(alpha, gamma)

# Learning curve plot (Average Qmax)
plotter.plot_q_vs_episodes(Ql_sarsa, "SARSA", "No. of episodes", "Average $Q_{max}$", "Gridworld, " + title, 'r')
plt.savefig("figures/gridworld/SARSA_learning_curve.png", dpi=400)

# Learning curve plot (Return)
plotter.plot_q_vs_episodes(Gl_sarsa, "SARSA", "No. of episodes", "Returns $G$", "Gridworld, " + title, 'r')
plt.savefig("figures/gridworld/SARSA_learning_curve_g.png", dpi=400)

# State value function plot
plotter.plot_q_vs_episodes(V_sarsa, "SARSA", "State, s", r"$V(s)$", r"Gridworld, $V^{*}(s)$, TD(0), " + title, 'r')
plt.savefig("figures/gridworld/SARSA_state_value.png", dpi=400)
# Policy plot
plotter.plot_q_vs_episodes(P_star_sarsa, "SARSA", "State, s", "$\pi(s)$", r"Gridworld, $\pi^{*}$, " + title, 'r')
plt.savefig("figures/gridworld/SARSA_policy.png", dpi=400)


# Plot policy and trajectory
plt.figure(figsize = (10, 8))
log = plotter.plot_trajectory(env, P_star_sarsa)
plt.plot(log['t'], log['s'], '-o')
plt.plot(log['t'][:-1], log['a'])
plt.plot(log['t'][:-1], log['r'])
plt.title("SARSA, Gridworld, "  + r"$\alpha = {}$, $\gamma = {}$".format(alpha, gamma), fontsize = 25)
plt.grid()
plt.legend(['s', 'a', 'r'])
plt.savefig('figures/gridworld/gridworld_sarsa_policy_trajec.png')

# Epsilon plots
val_name   = r"$\epsilon = $"
fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.set_figheight(15)
fig.set_figwidth(20)
fig.suptitle("SARSA, Gridworld, "  + r"$\alpha = {}$, $\gamma = {}$".format(alpha, gamma), fontsize = 25)
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
fig.savefig("figures/gridworld/grid_sarsa_epsilon.png", dpi=400)
plt.show()


# Alpha plots
val_name   = r"$\alpha = $"
fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.set_figheight(15)
fig.set_figwidth(20)
fig.suptitle("SARSA, Gridworld, "  + r"$\epsilon = \frac{1}{t}$, " + "$\gamma = {}$".format(gamma), fontsize = 25)
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
fig.savefig("figures/gridworld/grid_sarsa_alpha.png", dpi=400)
plt.show()



##########################
#       Q_LEARNING       #
##########################

Q_ql, Ql_qlearn, Gl_qlearn  = algorithms.Q_learning(env, total_eps, epsilon, True, alpha, gamma, Ql_qlearn, Gl_qlearn)
P_star_ql         = np.argmax(Q_ql, axis = 1)
print("P_star_QL\n", P_star_ql.reshape(5,5))
V_qlearn          = algorithms.TD0(env, P_star_ql, alpha, gamma, total_eps)
title             = r"$\epsilon = \frac{1}{t}$, " + r"$\alpha = {}, \gamma = {}$".format(alpha, gamma)

#Learning curve plot
plotter.plot_q_vs_episodes(Ql_qlearn, "Q-learning", "No. of episodes", "Average $Q_{max}$", "Gridworld, " + title, 'k')
plt.savefig("figures/gridworld/ql_learning_curve.png", dpi=400)

#Learning curve g plot
plotter.plot_q_vs_episodes(Gl_qlearn, "Q-learning", "No. of episodes", "Returns $G$", "Gridworld, " + title, 'k')
plt.savefig("figures/gridworld/ql_learning_curve_g.png", dpi=400)

# State value function plot
plotter.plot_q_vs_episodes(V_qlearn, "Q-learning", "State, s", r"$V(s)$", r"Gridworld, $V^{*}(s)$, TD(0), " + title, 'r')
plt.savefig("figures/gridworld/ql_state_value.png", dpi=400)
# Policy plot
plotter.plot_q_vs_episodes(P_star_ql, "Q-learning", "State, s", "$\pi(s)$", r"Gridworld, $\pi^{*}$, " + title, 'r')
plt.savefig("figures/gridworld/ql_policy.png", dpi=400)

# Plot policy and trajectory
plt.figure(figsize = (10, 8))
log = plotter.plot_trajectory(env, P_star_ql)
plt.plot(log['t'], log['s'], '-o')
plt.plot(log['t'][:-1], log['a'])
plt.plot(log['t'][:-1], log['r'])
plt.title("Q-Learning, Gridworld, "  + r"$\alpha = {}$, $\gamma = {}$".format(alpha, gamma), fontsize = 20)
plt.grid()
plt.legend(['s', 'a', 'r'])
plt.savefig('figures/gridworld/grid_qlearning_policy_trajectory.png')

# Epsilon plots
val_name   = r"$\epsilon = $"
fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.set_figheight(15)
fig.set_figwidth(20)
fig.suptitle("Q-learning, Gridworld, "  + r"$\alpha = {}$, $\gamma = {}$".format(alpha, gamma), fontsize = 25)
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
fig.savefig("figures/gridworld/grid_qlearning_epsilon.png", dpi=400)
plt.show()


# Alpha plots
val_name   = r"$\alpha = $"
fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.set_figheight(15)
fig.set_figwidth(20)
fig.suptitle("Q-learning, Gridworld, "  + r"$\epsilon = \frac{1}{t}$, " + "$\gamma = {}$".format(gamma), fontsize = 25)
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
fig.savefig("figures/gridworld/grid_qlearning_alpha.png", dpi=400)
plt.show()


