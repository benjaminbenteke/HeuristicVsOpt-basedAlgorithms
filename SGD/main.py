import numpy as np
import random
from scipy.optimize import minimize
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import optuna
import json
import time
import tracemalloc
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import ScalarFormatter

seed= 50
np.random.seed(seed)
random.seed(seed)

# -----------------------
# Game setting
# -----------------------
def g1(x):
    return x[0]+x[1]+x[2]-1
def g2(x):
    return x[0]+x[1]+x[2]-1
def g3(x):
    return x[2]-x[1]-x[0]
def g(x,v):
    if v==0: return g1(x)
    elif v==1: return g2(x)
    elif v==2: return g3(x)
    else: raise ValueError("Invalid v")

# -----------------------
# Shadow optimization
# -----------------------
def shadow(x, tol=1e-4):
    a,b,c = x

    def obj1(x): return 0.5*(x-1)**2 - x*b
    def obj2(x): return 0.5*(x-1)**2 - a*x
    def obj3(x): return 0.5*(x-1)**2

    cons1 = [{'type':'ineq','fun': lambda x: 1-x-b-c}]
    x_opt = minimize(obj1, a, method='SLSQP', constraints=cons1, tol=tol).x[0]

    cons2 = [{'type':'ineq','fun': lambda x: 1-x-a-c}]
    y_opt = minimize(obj2, b, method='SLSQP', constraints=cons2, tol=tol).x[0]

    cons3 = [{'type':'ineq','fun': lambda x: 0 - x + a + b}]
    z_opt = minimize(obj3, c, method='SLSQP', constraints=cons3, bounds=[(0,None)], tol=tol).x[0]

    return np.array([x_opt, y_opt, z_opt])

# -----------------------
# Candidate actions
# -----------------------
def get_actions(delta, M=30, nv=1):
    if nv==2:
        theta = np.random.uniform(0,2*np.pi,M)
        r = delta*np.sqrt(np.random.uniform(0,1,M))
        return np.column_stack((r*np.cos(theta), r*np.sin(theta)))
    return np.random.uniform(-delta,delta,M)

# -----------------------
# Metaheuristic solver with parallel action evaluation
# -----------------------
def algo_par(x0, G_max=500, N=3, M=500, tau=1.0, delta_init=10.0, eps=5e-4, burn_in=15, run_type="single"):
    x = x0.copy()
    delta = delta_init
    x_history = [x.copy()]
    res_fit, Delta_list = [], [delta]
    action_dim = [1,1,1]
    best_fitness_values = [np.inf]*N

    for t in range(G_max):
        x_hat = shadow(x, tol=min(delta,eps))
        best_fitness_values_prev = best_fitness_values.copy()
        best_fitness_values, best_actions = [], []

        for v in range(N):
            x_copy = x.copy()
            a_v_list = get_actions(delta, M, nv=action_dim[v])

            # Parallel evaluation of actions
            def evaluate_action(a_v):
                x_copy_v = x_copy.copy()
                x_copy_v[v] += a_v
                c_v = g(x_copy_v,v)
                if c_v < max(eps, delta):
                    return np.linalg.norm(x_copy_v - x_hat), a_v
                return np.inf, 0.0

            fitness_actions = Parallel(n_jobs=-1)(delayed(evaluate_action)(a) for a in a_v_list)
            f_v_star, a_v_star = min(fitness_actions, key=lambda x: x[0])

            if f_v_star==np.inf:
                f_v_star = np.linalg.norm(x-x_hat)
                a_v_star = 0.0

            best_fitness_values.append(f_v_star)
            best_actions.append(a_v_star)

        # Update x
        f_star_index = np.argmin(best_fitness_values)
        x[f_star_index] += best_actions[f_star_index]

        # Adaptive delta
        if t>burn_in:
            b_star= np.max(np.array(best_fitness_values_prev) - np.array(best_fitness_values))

            if b_star<0:
              delta= delta*2
            elif b_star<eps and np.array(best_actions).all() != 0.0:
              delta= delta
            
            elif b_star<eps and np.array(best_actions).all() == 0.0:
              delta= delta*2

            elif b_star>tau*eps:
              delta /= 2
            # b_star = np.max(np.array(best_fitness_values_prev)-np.array(best_fitness_values))
            # if b_star<0 or (b_star<eps and np.all(np.array(best_actions)==0.0)):
            #     delta *= 2
            # elif b_star>tau*eps:
            #     delta /= 2

        res_fit.append(np.linalg.norm(best_fitness_values, ord=np.inf))
        Delta_list.append(delta)
        x_history.append(x.copy())

        if res_fit[-1]<=eps: break

    if run_type=="tune":
        return (x if res_fit[-1]<=eps else None,
                {"iterations": t+1, "res_fit": res_fit, "deltas": Delta_list, "x_history": np.array(x_history)})
    elif run_type=="single":
        return x if res_fit[-1]<=eps else None, res_fit, Delta_list
    else:
        return x if res_fit[-1]<=eps else None

# -----------------------
# Generate feasible points
# -----------------------
def generate_points(n_pts=1000):
    s = np.random.rand(n_pts)
    t = np.random.rand(n_pts)
    x1 = t*s
    x2 = (1-t)*s
    x3 = np.random.rand(n_pts)*np.minimum(s,1-s)
    return np.column_stack((x1,x2,x3))

# -----------------------
# Single solver wrapper
# -----------------------

def run_single_point(x0, hyperparams):
    x,res_fit,Delta_list = algo_par(x0, G_max=500, N=3, M=hyperparams["M"],
                                    tau=hyperparams["tau"], delta_init=hyperparams["delta"],
                                    eps=5e-4, burn_in=hyperparams["burn_in"], run_type="single")
    return {"init_point": x0, "solution": x, "res_fit": res_fit, "Delta_list": Delta_list, "hit": x is not None}

# -----------------------
# Run experiments in parallel
# -----------------------
def run_experiments(init_points,hyperparams):
    # seeds = np.arange(len(init_points))+123
    return Parallel(n_jobs=-1)(delayed(run_single_point)(init_points[i],hyperparams) for i in range(len(init_points)))

# -----------------------
# JSON encoder
# -----------------------
class NumpyEncoder(json.JSONEncoder):
    def default(self,obj):
        if isinstance(obj,np.ndarray): return obj.tolist()
        return super().default(obj)

# -----------------------
# Main execution
# -----------------------
if __name__=="__main__":
    xx0 = generate_points(1000)
    x0 = xx0[0]

    # -----------------------
    # Hyperparameter tuning
    # -----------------------
    M_values=[500,1000]
    def objective(trial):
        M = trial.suggest_categorical("M", M_values)
        delta = trial.suggest_float("delta",1.0,20.0)
        tau = trial.suggest_float("tau",0.0,20.0)
        burn_in = trial.suggest_int("burn_in",2,20)
        x_final, info = algo_par(x0, M=M, delta_init=delta, tau=tau, burn_in=burn_in, run_type="tune")
        return info["res_fit"][-1]

    study = optuna.create_study(direction="minimize")
    # study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)) #
    study.optimize(objective, n_trials=100)
    hyperparams = study.best_params
    print("Best hyperparameters:", hyperparams)
    M, tau, i, delta = hyperparams["M"], hyperparams["tau"], hyperparams["burn_in"], hyperparams["delta"]
    

    with open("ex4_non_shared_best_hyperparams.json","w") as f:
        json.dump(hyperparams,f,indent=4)

    # Single test run with timing & memory outside
    # -----------------------
    start_time = time.time()
    tracemalloc.start()
    res = run_single_point(x0, hyperparams)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_time = time.time()
    print("Single hit:", res["hit"])
    print(f"Execution time: {end_time-start_time:.2f}s")
    print(f"Memory usage: current={current/1e6:.2f} MB; peak={peak/1e6:.2f} MB")

    # -----------------------
    # Run experiments
    # -----------------------
    results = run_experiments(xx0, hyperparams)
    with open("resultsEx4-non-shared.json","w") as f:
        json.dump(results,f,indent=2,cls=NumpyEncoder)

    # -----------------------
    # Collect statistics
    # -----------------------
    success_results = [r for r in results if r["hit"]]
    solutions = np.array([r["solution"] for r in success_results])
    print(f"Success rate: {len(success_results)/len(results)*100:.1f}%")
    np.savetxt("sol_ex4_non-shared.txt", solutions)

    
    import numpy as np

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(solutions[:,0], solutions[:,1], solutions[:,2],
            marker='o', s=40, edgecolor='black', facecolor='black',
            label="GNEs")
    ax.set_xlabel(r'$x_1$', fontsize=12, labelpad=10)
    ax.set_ylabel(r'$x_2$', fontsize=12, labelpad=10)
    ax.set_zlabel(r'$x_3$', fontsize=12, labelpad=15)

    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.zaxis.set_major_formatter(ScalarFormatter())

    x_min = solutions[:,0].min() 
    x_max = solutions[:,0].max() 
    y_min = solutions[:,1].min() 
    y_max = solutions[:,1].max()
    z_min = solutions[:,2].min()
    z_max = solutions[:,2].max()

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
    x_mid = (x_max + x_min) / 2.0
    y_mid = (y_max + y_min) / 2.0
    z_mid = (z_max + z_min) / 2.0

    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)


    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()

    plt.savefig('Ex4_non-shared.png', dpi=300, bbox_inches='tight', pad_inches=0.1)

    plt.show()

    # -----------------------
    # Hyperparameter summary
    # -----------------------
    print("Best final fitness:", study.best_value)






# # -*- coding: utf-8 -*-
# from math import *
# import numpy as np
# import random
# import matplotlib.pyplot as plt

# from scipy.optimize import minimize
# from joblib import Parallel, delayed
# np.random.seed(50)
# random.seed(50)
# from joblib import Parallel, delayed

# # Game setting

# def g1(x):
#   return x[0]+x[1]+x[2]-1

# def g2(x):
#   return x[0]+x[1]+x[2]-1

# def g3(x):
#   return x[2]-x[1]-x[0]

# def g(x, v):
#     if v == 0:
#         return g1(x)
#     elif v == 1:
#         return g2(x)
#     elif v == 2:
#         return g3(x)
#     else:
#         raise ValueError("Invalid value of v. Must be 0, 1, or 2.")


# def shadow(x, tol= 1e-4):
#     a, b, c= x
#     # Player 1 Objective
#     def obj1(x):
#       return 0.5*(x-1)**2-x*b

#     # Player 2 Objective
#     def obj2(x):
#         return 0.5*(x-1)**2-a*x

#     # Player 2 Objective
#     def obj3(x):
#         return 0.5*(x-1)**2

#     # Constraints for Player 1
#     def con1(x):
#         return 1 - x - b- c

#     # Player 1
#     x0 = a
#     cons1 = [{'type': 'ineq', 'fun': con1}]
#     solution = minimize(obj1, x0, method='SLSQP', constraints=cons1, tol= tol)

#     x_opt = solution.x[0]

#     # Player 2
#     def con2(x):
#         return 1 - x - a- c

#     # Initial guess
#     y0 = b


#     cons2 = [{'type': 'ineq', 'fun': con2}]
#     solution2 = minimize(obj2, y0, method='SLSQP', constraints=cons2, tol= tol)
#     y_opt = solution2.x[0] 

#     # Player 3
#     def con3(x):
#         return 0 - x + a + b

#     # Initial guess
#     z0 = c

#     b2 = [(0.0, None)] 

#     cons3 = [{'type': 'ineq', 'fun': con3}]
#     solution3 = minimize(obj3, z0, method='SLSQP', constraints=cons3, tol= tol, bounds=b2)

#     z_opt = solution3.x[0] 


#     return np.array([x_opt, y_opt, z_opt])



# def optimize_x_y_z(input_list):
#     """Run optimization in parallel."""
#     results = Parallel(n_jobs=-1, backend='loky')(delayed(shadow)((x, y, z)) for x, y, z in input_list)
#     return np.array(results)

# def get_actions(delta, M= 30, nv= 1):

#   if nv==2:
#     theta = np.random.uniform(0, 2*np.pi, M)
#     r = delta * np.sqrt(np.random.uniform(0, 1, M))
#     x = r * np.cos(theta)
#     y = r * np.sin(theta)
#     return np.column_stack((x, y))
#   elif nv==1:
#     return np.random.uniform(-delta, delta, M)


# """# SA"""

# G_max= 500
# n= 3
# N= 3
# M= 200 # Number of actions

# def algo_par(x0: np.ndarray,
#              G_max=500,
#              N=N,
#              M=500,
#              tau=1.0,
#              delta_init=10.0,
#              eps=5e-3,
#              burn_in=15,
#              debug=False, run_type="tune"):


#   # delta= 10.0 # Ball radius
#   action_dim= [1,1,1]
#   n = len(x0)
#   delta = delta_init
#   x = x0.copy()
#   x_history = np.zeros((G_max+1, n))
#   x_history[0] = x
#   res_fit, Delta_list = [], [delta]


#   best_fitness_values= []

#   for t in range(G_max):

#     # Get shadow
#     x_hat= shadow(x, tol= min(delta, eps))

#     best_fitness_values_prev= best_fitness_values.copy()

#     best_fitness_values= []
#     best_actions= []
    
    
#     for v in range(N):

#       x_copy= x.copy()
#       f_v_star, a_v_star= np.inf, None

#       a_v_list= get_actions(delta= delta, M= M, nv= action_dim[v])


#       for a_v in a_v_list:

#         x_copy[v]= x[v]+a_v
#         c_v= g(x_copy, v)


#         if c_v<max(eps, delta):


#           f_v= np.linalg.norm((x_copy-x_hat))

#           if f_v<f_v_star:
#             f_v_star= f_v
#             a_v_star= a_v

      
#       if f_v_star== np.inf:
#           f_v_star= np.linalg.norm((x-x_hat))
#           a_v_star= 0.0


#       best_fitness_values.append(f_v_star)
#       best_actions.append(a_v_star)

  

#     f_star_index= np.argmin(best_fitness_values)
#     a_star= best_actions[f_star_index]

#     if f_star_index==0:

#       x[0]= x[0]+a_star

#     elif f_star_index==1:
#       x[1]= x[1]+a_star

#     elif f_star_index==2:
#       x[2]= x[2]+a_star


#     if t>burn_in:
      # b_star= np.max(np.array(best_fitness_values_prev) - np.array(best_fitness_values))

      # if b_star<0:
      #   delta= delta*2
      # elif b_star<eps and np.array(best_actions).all() != 0.0:
      #   delta= delta
      
      # elif b_star<eps and np.array(best_actions).all() == 0.0:
      #   delta= delta*2

      # elif b_star>tau*eps:
      #   delta /= 2
        

#     res_fit.append(np.linalg.norm(best_fitness_values))
#     Delta_list.append(delta)
#     x_history[t+1] = x

#     if res_fit[-1] <= eps:
#         break

#   if run_type=="tune":
#     return (x if res_fit[-1] <= eps else None,
#           {"iterations": t+1, "res_fit": res_fit, "deltas": Delta_list, "x_history": x_history[:t+2]})
#   elif run_type=="single":
     
#      if t== G_max-1:
#         return None, res_fit, Delta_list
#      return x, res_fit, Delta_list
#   elif run_type=="many":
#      if t== G_max-1:
#         return None
#      return x

# print("********* Hyperparam tuning *********")
# # !pip install optuna


# M_values= [500,1000]
# n_pts = 1000

# # Sample s = x1 + x2 in [0,1]
# s = np.random.rand(n_pts)

# # Split s into x1, x2
# t = np.random.rand(n_pts)
# x1 = t * s
# x2 = (1 - t) * s

# # Sample x3 uniformly in [0, min(s, 1-s)]
# x3_upper = np.minimum(s, 1 - s)
# x3 = np.random.rand(n_pts) * x3_upper

# # Concatenate into shape (1000, 3)
# xx0 = np.column_stack((x1, x2, x3))
# # xx0= np.random.random((1000, n))*2
# x0= xx0[0]
# eps= 5*1e-4

# import optuna

# def objective(trial):
#     M = trial.suggest_categorical("M", M_values)
#     delta = trial.suggest_float("delta", 1.0, 20.0)
#     tau = trial.suggest_float("tau", 0.0, 20.0)
#     # eps = trial.suggest_float("eps", 1e-4, 5e-3, log=True)
#     burn_in = trial.suggest_int("burn_in", 2, 20)

#     # x0 = np.array([0.0,0.0])
#     x_final, info = algo_par(
#         x0,
#         M=M,
#         delta_init=delta,
#         tau=tau,
#         eps=eps,
#         burn_in=burn_in
#     )
#     return info["res_fit"][-1]

# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=100)


# # Running
# print("********* Running *********")

# G_max= 500

# optuna_res=  study.best_params #{'M': 500, 'delta': 3.0204635394778627, 'tau': 4.0894890477349755, 'burn_in': 9} #
# M= optuna_res["M"] # Number of actions

# tau= optuna_res["tau"]
# i= optuna_res["burn_in"]
# delta= optuna_res["delta"] # Ball radius


# for x0 in xx0:
#   x, res_fit, Delta_list= algo_par(x0,
#              G_max=500,
#              N=N,
#              M=M,
#              tau=tau,
#              delta_init=delta,
#              eps=eps,
#              burn_in=i,
#              debug=False, run_type= "single")
#   print("GNE: ", x)
#   if x is not None:
    
#     res_fit= np.array(res_fit)
#     np.savetxt('./fitness_ex4_non-shared.txt', res_fit)

#     Delta_list= np.array(Delta_list)
#     np.savetxt('./Delta_v_ex4_non-shared.txt', Delta_list)
#     break
#   else:
#     x, res_fit, Delta_list= algo_par(x0,
#              G_max=500,
#              N=N,
#              M=M,
#              tau=tau,
#              delta_init=delta,
#              eps=eps,
#              burn_in=i,
#              debug=False, run_type= "single")
#     res_fit= np.array(res_fit)
#     np.savetxt('./fitness_ex4_non-shared.txt', res_fit)
#     print("GNE: ", x)
#     Delta_list= np.array(Delta_list)
#     np.savetxt('./Delta_v_ex4_non-shared.txt', Delta_list)
#     break




# results = Parallel(n_jobs=-1, backend='loky')(delayed(algo_par)(x0,
#              G_max=500,
#              N=N,
#              M=M,
#              tau=tau,
#              delta_init=delta,
#              eps=eps,
#              burn_in=i,
#              debug=False, run_type= "many") for x0 in xx0)

# # Count None
# none_count = sum(x is None for x in results)

# # Filter non-None arrays
# non_none_arrays = [x for x in results if x is not None]

# # Convert to numpy array for easier handling
# non_none_array = np.vstack(non_none_arrays)

# print("Number of None:", none_count)
# print("Non-None arrays shape:", non_none_array.shape)
# print(non_none_array)

# results= np.array(non_none_array)

# np.savetxt('sol_ex4_non-shared.txt', results)

# ## Many experiments with different seed values

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.ticker import ScalarFormatter

# # -------------------------
# # Parametric curve
# # -------------------------
# alpha = np.linspace(0.0, 0.75, 50)  # 50 points from 0 to 3/4
# x = alpha
# y = 0.5 - alpha
# z = 0.5 * np.ones_like(alpha)

# # Example GNE results (replace with your actual results array)
# # results = np.load("results.npy")   # for instance
# # results = np.array([
# #     [0.0, 0.5, 0.5],
# #     [0.25, 0.25, 0.5],
# #     [0.5, 0.0, 0.5]
# # ])  # dummy data for demonstration

# # -------------------------
# # Create figure
# # -------------------------

# import numpy as np

# # Load a simple numeric array
# # results = np.loadtxt("/content/sol_ex4_non-share.txt")

# # print(results)
# # print(type(results))

# fig = plt.figure(figsize=(5, 4))
# ax = fig.add_subplot(111, projection='3d')

# # Scatter plot: parametric curve
# # ax.scatter(x, y, z,
# #            marker='o', s=40, edgecolor='red', facecolor='red')

# # Scatter plot: GNE results
# ax.scatter(results[:,0], results[:,1], results[:,2],
#            marker='o', s=40, edgecolor='black', facecolor='black',
#            label="GNEs")

# # -------------------------
# # Axis labels
# # -------------------------
# ax.set_xlabel(r'$x_1$', fontsize=12, labelpad=10)
# ax.set_ylabel(r'$x_2$', fontsize=12, labelpad=10)
# ax.set_zlabel(r'$x_3$', fontsize=12, labelpad=15)

# # No scientific notation
# ax.xaxis.set_major_formatter(ScalarFormatter())
# ax.yaxis.set_major_formatter(ScalarFormatter())
# ax.zaxis.set_major_formatter(ScalarFormatter())

# # -------------------------
# # Equal scaling for all axes (cube-like)
# # -------------------------

# # x_min = min(x.min(), results[:,0].min())
# # x_max = max(x.max(), results[:,0].max())
# # y_min = min(y.min(), results[:,1].min())
# # y_max = max(y.max(), results[:,1].max())
# # z_min = min(z.min(), results[:,2].min())
# # z_max = max(z.max(), results[:,2].max())

# x_min = results[:,0].min() #min(x.min(), results[:,0].min())
# x_max = results[:,0].max() #max(x.max(), results[:,0].max())
# y_min = results[:,1].min() #min(y.min(), results[:,1].min())
# y_max = results[:,1].max() #max(y.max(), results[:,1].max())
# z_min = results[:,2].min() #min(z.min(), results[:,2].min())
# z_max = results[:,2].max() #max(z.max(), results[:,2].max())

# max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
# x_mid = (x_max + x_min) / 2.0
# y_mid = (y_max + y_min) / 2.0
# z_mid = (z_max + z_min) / 2.0

# ax.set_xlim(x_mid - max_range, x_mid + max_range)
# ax.set_ylim(y_mid - max_range, y_mid + max_range)
# ax.set_zlim(z_mid - max_range, z_mid + max_range)

# # -------------------------
# # Legend & layout
# # -------------------------
# ax.legend(frameon=False, fontsize=10)
# fig.tight_layout()

# # Save high-resolution figure
# plt.savefig('Ex4_non-shared.png', dpi=300, bbox_inches='tight', pad_inches=0.1)

# plt.show()


# # === Example usage ===
# import numpy as np
# import multiprocessing as mp

# # === Your solver (metaheuristic) ===
# def run_solver(x0, seed):
#     #np.random.seed(seed)

#     x,res_fit,Delta_list= algo_par(x0,
#              G_max=500,
#              N=N,
#              M=M,
#              tau=tau,
#              delta_init=delta,
#              eps=eps,
#              burn_in=i,
#              debug=False, run_type= "single")

#     if x is not None:
#         return {
#                 "init_point": x0,
#                 "seed": seed,
#                 "T_eps": len(res_fit)+1,
#                 "fitness_curve": res_fit,
#                 "hit": True
#             }
#     else:
#         return {
#                 "init_point": x0,
#                 "seed": seed,
#                 "T_eps": len(res_fit)+1,
#                 "fitness_curve": res_fit,
#                 "hit": False
#             }

# print("********* many points *********")
# # === Run experiments in parallel with joblib ===
# def run_experiments(init_points, n_jobs=-1):
#     seeds = np.arange(len(init_points)) + 123  # deterministic unique seeds
#     results = Parallel(n_jobs=n_jobs)(delayed(run_solver)(init_points[i], seeds[i]) for i in range(len(init_points))
#     )
#     return results

# # === Example usage ===
# # xx0= np.random.random((1000, n))*2
# results = run_experiments(xx0)

# # === Collect statistics ===
# T_eps_vals = [r["T_eps"] for r in results if r["hit"]]
# success_rate = len(T_eps_vals) / len(results)

# print(f"Success rate: {success_rate*100:.1f}%")
# print(f"Mean T_eps: {np.mean(T_eps_vals):.1f}")
# print(f"Median T_eps: {np.median(T_eps_vals):.1f}")
# print(f"Std T_eps: {np.std(T_eps_vals):.1f}")

# import numpy as np
# import multiprocessing as mp



# import json

# class NumpyEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         if isinstance(obj, (np.integer, np.int64)):
#             return int(obj)
#         if isinstance(obj, (np.floating, np.float64)):
#             return float(obj)
#         return super(NumpyEncoder, self).default(obj)

# # Save to JSON file
# with open('resultsEx4-non-shared.json', 'w') as f:
#     json.dump(results, f, indent=2, cls=NumpyEncoder)

# # ------------------------
# # Best hyperparameters
# # ------------------------
# print("Best hyperparameters found:")
# print(study.best_params)
# print("Best final fitness:", study.best_value)

