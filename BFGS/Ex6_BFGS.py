




def backtracking_line_search(get_Theta, Grad_Theta, z, d, alpha=0.5, beta=0.5):
    """
    Backtracking line search to find an appropriate step size for a multivariate function.
    """
    t = 1.0  # Initial step size
    n_function_ev= 0

    grad_Phi_x = np.dot(Grad_Theta(z), d)


    # Backtracking line search
#     n_function_ev +=1
    while get_Theta(z + t * d) > get_Theta(z) + alpha * t * grad_Phi_x:
        t *= beta
        n_function_ev +=1
    return t,n_function_ev

def bfgs_optimization(x0):
    x = x0.copy()
    H = np.identity(len(x0))  # Initial approximation of the Hessian matrix
    V_list= []
    n_nev_total= 0
    k= 0
    
    while V(x)>sqrt(n+m)*eps:
#     for _ in range(500):
        V_list.append(V(x))
        # Evaluate the objective function and its gradient at the current solution
#         current_fitness = get_Theta(x)
        
        current_gradient = Grad_Theta(x)

        # Compute the descent direction using the inverse Hessian approximation
        d = -np.dot(H, current_gradient)

        # Perform backtracking line search to find the step size
#         step_size,n_nev = armijo_line_search(get_Theta, Grad_Theta, x, d)
        step_size,n_nev= backtracking_line_search(get_Theta, Grad_Theta, x, d)
        n_nev_total+= n_nev
#         print(step_size)
        # Update the current solution using BFGS update rule
        x_next = x + step_size * d

        # Compute the gradient difference for the BFGS update
        s = Grad_Theta(x_next) - Grad_Theta(x)
        y= x_next-x
        if np.isnan(s).any() or np.isnan(y).any():
#             print("Numerical instability detected. Exiting.")
            break
        
        H = H + np.outer((s - np.dot(H, s)), d) / (np.dot(d, s)+1e-8)
        x = x_next
        k+=1

    return x,V_list,n_nev_total,k


class TookTooLong(Warning):
    pass

class optimizer():
    def __init__(self):
        self.res=None
        self.n_fes= None
        self.n_iter= None
#         self.elapsed_time= elapsed_time

    def callback(self, x):
        
        self.res.append(V(x))
        if V(x)<=sqrt(n+m)*eps:
            warnings.warn("Terminating optimization: time limit reached",
                          TookTooLong)

    def objective_function_with_backtracking_line_search(self,z):
         """
         The scipy.minimize function does not directly support backtracking line search. You can still use custom step size searching methods like backtracking line search inside the objective function when defining your optimization problem. However, it won't affect the optimization algorithm itself, as scipy.minimize uses its own line search methods during the optimization process.
         
         """
         d= Grad_Theta(z)
#          search_d= -d # To use in backtracking_line_search: No need.
         step_size_,n_nev = backtracking_line_search(get_Theta, Grad_Theta, z, d, alpha, beta)
         return get_Theta(z + step_size_* d) # We do not need to use -d since in minimize, we've algready passed our gradient function which will be -grad inside.
    def optimize_single_initial_guess(self, z0):

        opt,V_list,n_nev_total,k = bfgs_optimization(z0)
        
        self.res= V_list
        self.n_fes= n_nev_total
        self.n_iter= k
        return opt

    def optimize(self, initial_guesses):
        # Run the optimization in parallel for all initial guesses
        num_cores = -1  # Set to the number of available CPU cores, or -1 to use all available cores
        results = Parallel(n_jobs=num_cores)(delayed(self.optimize_single_initial_guess)(z0) for z0 in initial_guesses)
        return results


import os

for num_points in range(100, 1501, 100):
    path = f"./Ex6/N_{num_points}"
    os.makedirs(path, exist_ok=True)
num_processes= -1
## Start
def f1(x):
  return -x[0]

def g11(x):
  return x[0]+x[1]-1


# p2 obejctive function and constraint
def f2(x):
  return -2*x[1]

def g21(x):
  return x[0]+x[1]-1



# Description
## P1
n1= 1 # num of decision variable controls by P1
m1= 1 # # num of inequality constraints for P1
g_1= [g11]

## P2
n2= 1 # num of decision variable controls by P2
m2= 1 # num of inequality constraints for P2
g_2= [g21]

# Concatening
n= n1+n2
m= m1+m2

x = np.random.uniform(0, 1, 100)
y = np.random.uniform(0, 1, 100)

mask = (x + y <= 1)

# Get the points that meet the constraints
x_filtered = x[mask][:1]
y_filtered = y[mask][:1]
lambd01, lambd02= np.array([[0.0]*m1]*len(x_filtered)),np.array([[0.0]*m2]*len(x_filtered))
lambd0= np.concatenate((lambd01,lambd02),1)
z0 = np.concatenate((np.expand_dims(x_filtered, axis=1),np.expand_dims(y_filtered, axis=1)), 1)
z0 = np.concatenate((z0,lambd0), 1).tolist()[0]

g= g_1+g_2
n_const= m

def F(z):
  x = z[:-n_const]
  if n_const>1:
    lambd = z[-n_const:]
  else:
    lambd = z[-n_const]
    lambd= [lambd]
  return [lambd[0]-1, lambd[1]-2]


N= 2 # Two players
mm= [m1, m2]

# To be adapted.
def Jacobiang(z):
  """
    You can pass z just for testing. But only x is the right argument.
    Return jacobian of g w.r.t. x
  """
  # return np.diag([1,1])
  x = z[:-n_const]
  if n_const>1:
    lambd = z[-n_const:]
  else:
    lambd = z[-n_const]
    lambd= [lambd]

  return np.array([[1,1],[1,1]])

def JacobianFLambda(z):
  """
    You can pass z just for testing. But lambd is the right argument.
    Return jacobian of F w.r.t. lambd
  """
  # x= np.array(z[:-m-m])
  # lambd= np.array(z[n:-m])
  # w= np.array(z[-m:])
  x = z[:-n_const]
  if n_const>1:
    lambd = z[-n_const:]
  else:
    lambd = z[-n_const]
    lambd= [lambd]
  # return np.array([[3.25,2.29115,0,0,0,0], [0,0,1.25,1.5625, 0,0], [0,0,0,0,4.125,2.8125]]) # Gradient of g
  return np.array([[1,0], [0,1]]) # Gradient of g

# To be adapted.
def JacobianF(z):
  """
    You can pass z just for testing. But only x is the right argument.
    Return jacobian of F w.r.t. x
  """
  x = z[:-n_const]
  if n_const>1:
    lambd = z[-n_const:]
  else:
    lambd = z[-n_const]
    lambd= [lambd]

  return np.array([[0,0],[0,0]])

n= 2
def run_parallel(num_points):
    lower_bnd, upper_bnd= 0.0, 1.0
    num_initial_points= num_points
    x0= np.random.uniform(lower_bnd, upper_bnd, size=(num_points, n))
    lambd01, lambd02= np.array([[0.0]*m1]*num_initial_points),np.array([[0.0]*m2]*num_points)
    g= g_1+g_2
    lambd0= np.concatenate((lambd01,lambd02),1)
    z0 = np.concatenate((x0,lambd0), 1)

    n_const= m
    op = optimizer()
    results = op.optimize(z0)

    solns= organize_sol(results)
    set_of_points= organize_solutions(solns)
    solns= organize_sol(results)
    solns= np.array(solns)[:,:2]
    return solns



nubmer_points_list= [100, 200,300,400,500,600,700,800,900,1000,1100,1200, 1300,1400,1500]
n_runs= [10]

def run_with_diff_n_runs(num_points):
    final_res= []
    for n_r in n_runs:
        temp_res= []
        
        for i in range(n_r):
            res= run_parallel(num_points)
            res= np.array(res)
            np.savetxt('./Ex6/N_'+str(num_points)+"/"+str(i+1)+"_"+"solns"+'_'+'run_'+str(n_r)+'_'+str(num_points)+'pts'+'.txt', res, delimiter=',')
                  
    return final_res

Parallel(n_jobs=num_processes)(delayed(run_with_diff_n_runs)(num_points) for num_points in nubmer_points_list)
