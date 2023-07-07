import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import quad
from RealIntegratedLuminosity import L_int_summary_18

plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Helvetica",
  "font.size": 12
})

f=open('Data/a_18_4Par.txt',"r")
lines=f.readlines()
a_18=[]
for x in lines:
    a_18.append(float(x.split(' ')[0]))  

f.close()

f=open('Data/b_18_4Par.txt',"r")
lines=f.readlines()
b_18=[]
for x in lines:
    b_18.append(float(x.split(' ')[0]))  

f.close()

f=open('Data/c_18_4Par.txt',"r")
lines=f.readlines()
c_18=[]
for x in lines:
    c_18.append(float(x.split(' ')[0]))  

f.close()

f=open('Data/d_18_4Par.txt',"r")
lines=f.readlines()
d_18=[]
for x in lines:
    d_18.append(float(x.split(' ')[0]))  

f.close()

f=open('Data/ts_18_4Par.txt',"r")
lines=f.readlines()
ts_18=[]
for x in lines:
    ts_18.append(float(x.split(' ')[0]))  

f.close()


#model parameters and initial guesses
ts_18=np.array(ts_18)
a_18=np.array(a_18)
b_18=np.array(b_18)
c_18=np.array(c_18)
d_18=np.array(d_18)


#Objective Function
def fun(t1):
    result=np.empty(len(x0))
    for i in range(len(x0)):
        lam=lambda x1: a_18[i]*np.exp(-(b_18[i]*x1))+c_18[i]*np.exp(-d_18[i]*x1)
        result[i]=-quad(lam, 0, t1[i])[0]
        
    result = np.sum(result)
    return result

#constraint
def cons(t1):
    res = np.sum(t1) - (tot)
    return res

#jacobian of the objective function
def jacb(t1):
    der=-a_18*np.exp(-(b_18*t1))-c_18*np.exp(-d_18*t1)
    #result=np.sum(der)
    return der
        
    
#Initial guesses    
x0=ts_18

#constraint determination
tot=sum(x0)

#bounds
list=[[1800,86400]]
for li in range(1, len(a_18)):
    list=list+[[1800, 86400]]
        
bnd=list

#optimization
res = minimize(fun, x0, options={'disp': True, 'maxiter':10000}, constraints={'type':'eq', 'fun': cons, 'jac': lambda x: np.ones(len(x0))}, jac=jacb, method='SLSQP', bounds=bnd) #      

#saving optimized times
with open('Data/res_opt_2018.txt', 'w') as f:
        f.write('')
        f.close()
for el in res.x:
    with open('Data/res_opt_2018.txt', 'a') as f:
        f.write(str(el))
        f.write('\n')


