# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 14:03:09 2021

@author: grego


"""
import time
import numpy as np
import scipy as sp
import matplotlib
#------------------------------------------------------------
import gen_data     
import ipm_func
import svm_func
from parameters import m,w,n,p,gamma,sigma_step,sigma,tol_cg, MAXIT_cg, DENSITY,power


def synthetic_run_params(m, w, n, p, gamma, sigma_step, sigma, tol_cg, MAXIT_cg, DENSITY, power):
    # Data LOAD 
    X_train, y_train, X_test, y_test, w_true = gen_data.gen_data_SVM(m,n,p,DENSITY)
    
    m,N = np.shape(X_train)
    n = 2*N+1

    # form l1-SVM constraint matrix 
    A,b,c = svm_func.formL1SVM(X_train,y_train)

    # IPM Standard
    t_stan_1 = time.time()
    x,y,s,iter_out_stan,iter_in_cg_vec_stan,kap_AD_vec,time_ls_stan = ipm_func.ipm_standard(m,n,A,b,c,tol_cg)
    t_stan_2 = time.time()
    # IPM Ours
    t_ipm_1 = time.time()
    x,y,s,iter_out_ipm,iter_in_cg_vec_ipm,kap_ADW_vec,v_vec,time_ls_ipm = ipm_func.ipm(m,n,w,A,b,c,tol_cg)
    t_ipm_2 = time.time()


    results = dict()
    results['ipm_time'] = t_ipm_2 - t_ipm_1   
    results['stan_time'] = t_stan_2 - t_stan_1
    results['niter_out_stan'] = iter_out_stan
    results['niter__out_ipm'] = iter_out_ipm
    results['niter_inner_stan'] = iter_in_cg_vec_stan
    results['niter_inner_cg'] = iter_in_cg_vec_ipm
    results['inner_cg_vec'] = iter_in_cg_vec_stan
    results['kap_AD_vec'] = kap_AD_vec
    results['kap_ADW_vec'] = kap_ADW_vec
    results['v_vec'] = v_vec

    return results
    

if __name__ == '__main__':
    
    print(synthetic_run_params(m,w,n,p,gamma,sigma_step,sigma,tol_cg, MAXIT_cg, DENSITY,power))
    






    
