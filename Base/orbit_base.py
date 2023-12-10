#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 18:21:07 2023

@author: renatotronofigueras
"""
import numpy as np
from Util.initialize_case import initialize_case

class OrbitBase:
    
    def __init__(self,lower_bounds,upper_bounds,A_n0,max_evaluations,t,kernel_type,w_r_pattern,poly):
        
        assert(len(lower_bounds) == len(upper_bounds)), "init: The lengths of the lower and upper bounds do not match"
        
        self.lower_bounds_ = lower_bounds #Lower bounds defining the hypercube
        self.upper_bounds_ = upper_bounds #Upper bounds defining the hypercube
        self.d_ = len(self.lower_bounds_) #Dimension of the space
        self.A_n_ = A_n0 #Set of evaluated points
        self.n_ = len(self.A_n_) #Number of evaluated points
        self.max_evaluations_ = max_evaluations #Maximum number of function evaluations allowed
        self.t_ = t #Size of the new set of points created in each iteration
        
        self.kernel_type_ = kernel_type #Dictionary with kernel type and related parameters
        self.kernel_func_ = None #Kernel function
        
        self.set_kernel()
        
        self.history_ = None
        
        self.poly_ = poly
        
        self.phi_ = None #Phi matrix
        self.p_ = None #P matrix
        self.F_ = None #F vector
        
        self.lambda_ = None #Coefficients for RBF's
        self.c_ = None #Coefficients for polynomial
        
        self.initialized_ = False
        
        self.fit_surrogate_model() #Set initial coefficients
        
        i_best = np.argmin(self.history_)
        self.best_point_ = self.A_n_[i_best]
        self.best_value_ = self.history_[i_best]
        
        self.w_r_pattern_ = w_r_pattern #Pattern for the weights
        self.N_ = len(self.w_r_pattern_) #Length of the weight cycle
        self.counter_ = 0 #Counter usfeul for keeping track of the pattern 
        
        self.omega_n_ = None
        
    def get_coefficients(self):
        
        return self.lambda_, self.c_
        
    def objective_function(self,x):
        
        raise NotImplementedError("The objective function must be defined in each derived class")
        
    def d_norm(self,u):
        
        assert(len(u) == self.d_), "d_norm: The argument is not d-dimensional" 
        
        return np.linalg.norm(u)
    
    def set_kernel(self):
        
        if self.kernel_type_['type'] == 'surface spline':
            
            assert('kappa' in self.kernel_type_), "set_kernel: Surface spline needs a value for kappa"
            
            kappa = self.kernel_type_['kappa']
            
            assert(isinstance(kappa, int) or (isinstance(kappa,float) and kappa.is_integer())), "set_kernel: kappa is not a natural number"
            
            if kappa % 2 == 0:
                
                def kernel_func(r):
                    
                    if r < 1e-14:
                        
                        r = 1e-12
                    
                    return r**(kappa)*np.log(r)
                
            else:
                
                def kernel_func(r):
                    
                    return r**(kappa)
                
        elif self.kernel_type_['type'] == 'multiquadratic' or self.kernel_type_['type'] == 'inverse multiquadratic':
            
            assert('kappa' in self.kernel_type_), "set_kernel: Multiquadratic and inverse multiquadratic need a value for kappa"
            
            kappa = self.kernel_type_['kappa']
            
            assert(not (isinstance(kappa, int) or (isinstance(kappa,float) and kappa.is_integer()))), "set_kernel: kappa is a natural number"
            
            assert('gamma' in self.kernel_type_), "set_kernel: Multiquadratics and inverse multiquadratics need a value for gamma"
                
            gamma = self.kernel_type_['gamma']
                
            if self.kernel_type_['type'] == 'multiquadratic':
                
                assert(kappa > 0), "set_kernel: Multiquadratic kernel needs a positive value for Gamma"
                
                def kernel_func(r):
                    
                    return (r**2 + gamma**2)**(kappa)
                
            else:
                
                assert(kappa < 0), "set_kernel: Multiquadratic kernel needs a negative value for Gamma"
                
                def kernel_func(r):
                    
                    return (r**2 + gamma**2)**(kappa)
                    
        elif self.kernel_type_['type'] == 'Gaussian':
            
            assert('gamma' in self.kernel_type_), "set_kernel: Gaussian kernel need a value for gamma"
            
            gamma = self.kernel_type_['gamma']
            
            assert(gamma > 0), "set_kernel: gamma must be positive"
            
            def kernel_func(r):
                
                return np.exp(-gamma*r**2)
            
        else:
            
            raise NotImplementedError("The kernel selected is not implemented")
            
        self.kernel_func_ = kernel_func
        
    def fit_surrogate_model(self):
        
        #Solve linear system to determine RBF coefficients
        if self.initialized_ == False:
            
            self.phi_ = np.zeros((self.n_,self.n_))
            self.F_ = np.zeros((self.n_))
            self.history_ = np.zeros(self.n_)
            
            for i in range(self.n_):
                
                x_i = self.A_n_[i]
                
                for j in range(i,self.n_):
                    
                    x_j = self.A_n_[j]
                    
                    r = self.d_norm(x_i - x_j)
                    
                    self.phi_[i,j] = self.kernel_func_(r)
                    self.phi_[j,i] = self.phi_[i,j]

                self.F_[i] = self.evaluate_objective_function(x_i)
                
                self.history_[i] = self.F_[i]
            
            if self.poly_:
                   
                self.p_ = np.zeros((self.n_,self.d_+1))
                    
                for i in range(self.n_):
                    
                    x_i = self.A_n_[i]
                    
                    for j in range(self.d_ + 1):
                    
                        if j == 0:
                            
                            self.p_[i,j] = 1
                
                        else:
                    
                            self.p_[i,j] = x_i[j-1]
                
                assert(np.linalg.matrix_rank(self.p_) == self.d_+1), "fit_surrogate_model: Singular P matrix at initialization"
            
            self.initialized_ = True
            
        else:
            
            self.phi_ = np.pad(self.phi_, ((0,1),(0,1)), mode = 'constant', constant_values = 0)
            
            i = self.n_ - 1
            
            x_i = self.A_n_[i]
            
            for j in range(self.n_):
                
                x_j = self.A_n_[j]
                
                r = self.d_norm(x_i - x_j)
                
                self.phi_[i,j] = self.kernel_func_(r)
                self.phi_[j,i] = self.phi_[i,j]
            
            self.F_ = np.append(self.F_, self.evaluate_objective_function(x_i))
            
            self.history_ = np.append(self.history_, self.F_[-1])
            
            if self.poly_:
            
                self.p_ = np.pad(self.p_, ((0,1),(0,0)), mode = 'constant', constant_values = 0)
            
                for j in range(self.d_ + 1):
                
                    if j == 0:
                        
                        self.p_[i,j] = 1
            
                    else:
                
                        self.p_[i,j] = x_i[j-1]
        
        if self.poly_:
            
            upper_left = self.phi_
            upper_right = self.p_  
            bottom_left = self.p_.T
            bottom_right = np.zeros((self.d_+1,self.d_+1))
              
            lhs_matrix = np.block([[upper_left,upper_right],[bottom_left,bottom_right]])
            
            rhs_vector = np.concatenate((self.F_,np.zeros((self.d_+1))))
            
            coefficients = np.linalg.solve(lhs_matrix,rhs_vector)
            
            self.lambda_, self.c_ = coefficients[0:self.n_], coefficients[self.n_:]
            
        else:
            
            self.lambda_ = np.linalg.solve(self.phi_,self.F_)
    
    def evaluate_surrogate_model(self,x):
        
        s = 0
        
        for i in range(self.n_):
            
            r = self.d_norm(self.A_n_[i] - x)
            
            s += self.lambda_[i]*self.kernel_func_(r)
        
        if self.poly_:
            
            for i in range(self.d_+1):
                
                if i == 0:
                    
                    s += self.c_[i]
                    
                else:
                    
                    s += self.c_[i]*x[i-1]
            
        return s
            
    def create_distance_matrix(self):

        D = np.linalg.norm(self.omega_n_[:, np.newaxis, :] - self.A_n_[np.newaxis, :, :], axis=2)
    
        return D
            
    def create_random_candidate_points(self):
        
        self.omega_n_ = np.random.rand(self.t_,self.d_)*(self.upper_bounds_ - self.lower_bounds_) + self.lower_bounds_
    
    def determine_best_candidate_point(self):
        
        s = np.array([self.evaluate_surrogate_model(x) for x in self.omega_n_])
        
        s_max = np.max(s)
        s_min = np.min(s)
        
        D = self.create_distance_matrix()
        
        min_distances = np.min(D, axis=1)
        min_distance_max = np.max(min_distances)
        min_distance_min = np.min(min_distances)
        
        if s_max != s_min:
            
            vr = (s - s_min) / (s_max - s_min)
        
        else:
            
            vr = np.ones_like(s)
    
        if min_distance_max != min_distance_min:
            
            vd = (min_distance_max - min_distances) / (min_distance_max - min_distance_min)
        
        else:
            
            vd = np.ones_like(min_distances)
            
        w_r = self.w_r_pattern_[self.counter_]

        w_d = 1 - w_r
        
        w = w_r*vr.T + w_d*vd
        
        return self.omega_n_[np.argmin(w)]
        
    def evaluate_objective_function(self,x):
        
        return self.objective_function(x)
    
    def single_step(self):
        
        best_value_ini = self.best_value_
        
        self.create_random_candidate_points()
        
        new_point = self.determine_best_candidate_point()
        
        self.A_n_ = np.vstack([self.A_n_,new_point])

        self.n_ = len(self.A_n_)
        
        self.fit_surrogate_model()
        
        i_best = np.argmin(self.history_)
        
        self.best_point_ = self.A_n_[i_best]
        
        self.best_value_ = self.history_[i_best]
        
        self.counter_ += 1
        
        if self.counter_ == self.N_:
            
            self.counter_ = 0
                
    def solve(self):
        
        while(self.n_ <= self.max_evaluations_):
            
            print(f"Evaluations: {self.n_}/{self.max_evaluations_}")
            self.single_step()
            
            print('Best value: ', self.best_value_)
            
        return self.best_point_, self.best_value_
    
    def get_current_solution(self):
        
        return self.best_point_, self.best_value_


            