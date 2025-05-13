import numpy as np
import numba as nb

from EconModel import jit
from consav.linear_interp import interp_2d, interp_1d_vec_mon_noprep

#########
# solve #
#########

@nb.njit
def inverse_marg_util(par,u):
	""" Inverse function of marginal utility of consumption """
	return 1/u

@nb.njit
def marg_util_con(par,c):
	""" marginal utility of consumption """
	return 1/c

@nb.njit
def utility(par,c):
	""" utility function """
	return np.log(c)

@nb.njit
def EGM(t,par,sol):
	""" EGM for policy functions at time t"""

	sol_con = sol.sol_con
	m_pd_grid = par.m_pd_grid
	m_grid = par.m_grid
	p_grid = par.p_grid

	# a. last period
	if t == par.T-1:
		for i_p in nb.prange(par.Np):
			sol_con[t,i_p,:] = m_grid

	# b. other periods
	else:
		shape = sol_con[0].shape
		q_grid = np.zeros(shape)

		# i. next-period marginal value of cash
		for i_p in range(par.Np):
			p = p_grid[i_p]
			for i_m_pd in range(par.Nm_pd):
				m_pd = m_pd_grid[i_m_pd]
				q = compute_q(par,sol,t,m_pd,p)
				q_grid[i_p,i_m_pd] = q

		# ii. endogenous grid and interpolation to common grid
		for i_p in range(par.Np):
			interp_to_common_grid(par,sol,t,q_grid,i_p)

@nb.njit
def compute_q(par,sol,t,m_pd,p):
	""" compute post-decision marginal value of cash """

	# unpack
	sol_con = sol.sol_con
	m_grid = par.m_grid
	p_grid = par.p_grid

	# a. initialize q
	q = 0.0

	# b. loop over psi and xi
	for i_psi in range(par.Npsi):
		for i_xi in range(par.Nxi):
			# o. nodes
			xi = par.xi[i_xi]
			psi = par.psi[i_psi]

			# oo. next-period states
			p_plus = p**(par.rho_p) * xi
			m_plus = (1+par.r)*m_pd + psi*p_plus 
			
			# ooo. next-period consumption and marginal utility
			c_plus = interp_2d(p_grid,m_grid,sol_con[t+1],p_plus,m_plus) # slice to get rid of sigma_xi and sigma_psi
			mu_plus = marg_util_con(par, c_plus)
			
			# oooo. add to q
			q += par.psi_w[i_psi]*par.xi_w[i_xi]*mu_plus

	return q
				
@nb.njit
def interp_to_common_grid(par,sol,t,q_grid,i_p):
	""" endogenous grid method """
	
	# o. temp
	m_grid = par.m_grid
	m_pd_grid = par.m_pd_grid
	sol_con = sol.sol_con

	m_temp = np.zeros(par.Nm_pd+1)
	c_temp = np.zeros(par.Nm_pd+1)
	
	# o. endogenous grid
	for i_m_pd in range(par.Nm_pd):
		m_pd = m_pd_grid[i_m_pd]
		q_index = (i_p, i_m_pd)
		c_temp[i_m_pd+1] = inverse_marg_util(par, par.beta*(1+par.r)*q_grid[q_index])
		m_temp[i_m_pd+1] = m_pd + c_temp[i_m_pd+1]

	# oo. interpolation to common grid
	# add index together
	sol_con_index = (t,i_p)
	interp_1d_vec_mon_noprep(m_temp,c_temp,m_grid,sol_con[sol_con_index])


##############
# simulation #
##############

@nb.njit(parallel=True)
def policy_interp(par,sol,N,t,states,con):
	""" interpolate policy function to get consumption given current states """
	for i in nb.prange(N):
		con[i]  = interp_2d(par.p_grid, par.m_grid, sol.sol_con[t],states[i,1],states[i,0])
		

def simulate(par,sol,sim):
	""" simulate model """

	# a. unpack
	states = sim.states # shape (T,N,Nstates)
	states_pd = sim.states_pd # shape (T,N,Nstates_pd)
	actions = sim.actions # shape (T,N,Nactions)
	outcomes = sim.outcomes # shape (T,N,Noutcomes)
	shocks = sim.shocks # shape (T,N,Nshocks)
	reward = sim.reward # shape (T,N,Nstates)

	m = states[:,:,0]
	p = states[:,:,1]

	m_pd = states_pd[:,:,0]
	p_pd = states_pd[:,:,1]

	savings_rate = actions[:,:,0]
	
	xi = shocks[:,:,0]
	psi = shocks[:,:,1]

	c = outcomes[:,:,0]
	rho_p = par.rho_p	
		
	# c. time loop  
	for t in range(par.T):
		# a. final period consumption
		if t == par.T-1:
			c[t] = m[t]
			
		# b. consumption all other periods
		else:			
			policy_interp(par,sol,sim.N,t,states[t],c[t])
			
		# c. reward
		reward[t] = utility(par,c[t])
		
		# d. post-decision states
		m_pd[t] = m[t]-c[t]
		p_pd[t] = p[t]

		# savings rate
		savings_rate[t] = m_pd[t]/m[t]

		# e. next period
		if t < par.T-1:
			# i. permanent income state
			p[t+1] = (p_pd[t]**rho_p)*xi[t+1]
			
			# ii. income
			y = p[t+1]*psi[t+1]
			
			# iii. cash-on-hand
			m[t+1] = (1+par.r)*m_pd[t] + y
