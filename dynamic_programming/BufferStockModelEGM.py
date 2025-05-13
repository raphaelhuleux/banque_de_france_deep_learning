import numpy as np
from EconModel import EconModelClass, jit
from consav.grids import nonlinspace # grids
from dynamic_programming.egm import EGM, simulate
from consav.quadrature import log_normal_gauss_hermite
import torch 

class BufferStockModelEGMClass(EconModelClass):

    def settings(self):
        """ basic settings """
        
        self.namespaces = ['par','sol','sim'] # must be numba-able
    

    def setup(self):
        """ choose parameters """

        par = self.par
        sim = self.sim

        par.seed = 1 # seed for random number generator
        torch.manual_seed(par.seed)

        # model parameters
        par.T = 5 # number of periods
        par.beta = 1.0/1.04 # discount factor
        par.r = 0.02 # return rate

        par.sigma_psi = 0.3 # shock, std
        par.sigma_xi = 0.1 
        par.rho_p = 0.95

        # initial states
        par.mu_m0 = 1.0 # mean of initial cash-on-hand
        par.sigma_m0 = 0.1 # std of initial cash-on-hand

        par.mu_p0 = 1.0 # mean of initial persistent productivity
        par.sigma_p0 = 0.1 # std of initial persistent productivity

        # b. egm
        par.Nm_pd = 1_000 # number of grid points
        par.Nm = 1_000 # number of grid points
        par.Np = 1_000 # number of grid points

        par.m_max = 10.0 # max cash-on-hand
        par.p_max = 10.0 # max permanent income
        par.p_min = 0.1 # min permanent income

        par.Nxi = 10 # number of quadrature nodes
        par.Npsi = 10 # number of quadrature nodes

        sim.N = 100_000

        
    def allocate(self):
        """ allocate arrays """

        par = self.par
        sim = self.sim
        sol = self.sol

        # grids 
        par.m_pd_grid = nonlinspace(0.0,par.m_max,par.Nm_pd,1.4)
        par.m_grid = nonlinspace(0.0,par.m_max,par.Nm,1.4)
        par.p_grid = nonlinspace(0.1,par.p_max,par.Np,1.1)

        par.psi, par.psi_w = log_normal_gauss_hermite(par.sigma_psi,par.Npsi)
        par.xi, par.xi_w = log_normal_gauss_hermite(par.sigma_xi,par.Nxi)

        # solution objects
        sol.sol_con = np.zeros((par.T, par.Np, par.Nm))

        # simulations 
        sim.states = np.zeros((par.T,sim.N,2)) # state-vector
        sim.states_pd = np.zeros((par.T,sim.N,2)) # post-decision state vector
        sim.shocks = np.zeros((par.T,sim.N,2)) # shock-vector
        sim.outcomes = np.zeros((par.T,sim.N,2)) # outcomes array
        sim.actions = np.zeros((par.T,sim.N,1))  # actions array
        sim.reward = np.zeros((par.T,sim.N)) # array for utility rewards
        sim.R = np.nan # initialize average discounted utility
                
    def draw_initial_states(model,N):
        """ draw initial state (m,p,t) """
    
        par = model.par
        
        # draw initial cash-on-hand
        m0 = par.mu_m0*torch.exp(torch.normal(-0.5*par.sigma_m0**2,par.sigma_m0,size=(N,))).cpu().numpy().astype(np.float64)
        
        # draw permanent income
        p0 = par.mu_p0*torch.exp(torch.normal(-0.5*par.sigma_p0**2,par.sigma_p0,size=(N,))).cpu().numpy().astype(np.float64)
        return np.stack((m0,p0),axis=1) # (N,Nstates)

    def draw_shocks(model,N):
        """ draw shocks """

        par = model.par

        xi = torch.normal(0.0,1.0,size=(par.T,N))
        xi = torch.exp(par.sigma_xi*xi-0.5*par.sigma_xi**2).cpu().numpy().astype(np.float64)

        psi = torch.normal(0.0,1.0,size=(par.T,N))
        psi = torch.exp(par.sigma_psi*psi-0.5*par.sigma_psi**2).cpu().numpy().astype(np.float64)
        return np.stack((xi,psi),axis=-1) # (T,N,Nshocks)
        
    def solve_EGM(self):
        """ solve with EGM """

        with jit(self) as model:
            par = model.par
            sol = model.sol
            for t in reversed(range(par.T)):
                EGM(t,par,sol)
            
    ############
    # simulate #
    ############

    def simulate_R(self):	
        """ simulate life time reward """

        par = self.par
        sim = self.sim

        sim.states[0] = self.draw_initial_states(sim.N)
        sim.shocks = self.draw_shocks(sim.N)

        # a. simulate
        with jit(self) as model:
            simulate(model.par,model.sol,model.sim)

        # b. compute R
        beta = par.beta 
        beta_t = np.zeros((par.T,sim.N))
        for t in range(par.T):
            beta_t[t] = beta**t

        sim.R = np.sum(beta_t*sim.reward)/sim.N
           