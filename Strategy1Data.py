import numpy as np
import matplotlib.pyplot as plt
import LoadData as ld
import scipy.integrate as integrate
from lmfit import Model
import matplotlib.ticker as mticker

plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Helvetica",
  "font.size": 12
})


#loading data
data_16, data_17, data_18, array16, array17, array18 = ld.Data()
data_tot, dataTot, arrayTot = ld.TotalDataSet(data_16, data_17, data_18)
data_ta16, data_tf16, data_ta17, data_tf17, data_ta18, data_tf18 = ld.loadFill()     
FillNumber16, FillNumber17, FillNumber18 = ld.FillNumber()
data_16_sec, data_ta16_sec, data_tf16_sec, data_17_sec, data_ta17_sec,\
   data_tf17_sec, data_18_sec, data_ta18_sec, data_tf18_sec=ld.Data_sec(array16,\
      data_ta16, data_tf16, array17, data_ta17, data_tf17, array18, data_ta18, data_tf18)
   

with open('Data/L30m16.txt', 'w') as f:
  f.write('')
  f.close()

for i in range(len(FillNumber16)):
    #plotting results 
    plt.close("all")
    text = str(int(FillNumber16[i])) #number of current fill
    #obtain the Times and luminosity evolution values for that fill
    f=open('ATLAS/ATLAS_fill_2016/{}_lumi_ATLAS.txt'.format(text),"r")
    lines=f.readlines()
    L_evolx=[]
    times=[]
    for x in lines:
        times.append(int(x.split(' ')[0]))  
        L_evolx.append(float(x.split(' ')[2]))
          
    f.close()
    Times = np.array(times)
    L_evol = np.array(L_evolx)
    
    
    #deleting the null values of the luminosity
    zero=np.where(L_evol<100)
    L_zero=np.delete(L_evol, zero)
    T_zero=np.delete(Times, zero)
        
    #check for enough points
    if len(L_zero)<10:
        zero=np.where(L_evol<5)
        L_zero=np.delete(L_evol, zero)
        T_zero=np.delete(Times, zero)

    #defining the derivative 
    dy = np.zeros(L_zero.shape)
    dy[0:-1] = np.diff(L_zero)/np.diff(T_zero)

     #start to slim down the fit interval       
    L_tofit=[]
    T_tofit=[]
    for idx in range(len(L_zero)):
        #cancelling too strong derivative points
        if dy[idx]<0 and dy[idx]>-1.5:
            L_tofit.append(L_zero[idx])
            T_tofit.append(T_zero[idx])
        if dy[idx]>0 or dy[idx]<-1.5:
            continue     
        
    #evaluating the differences between two subsequent points
    diff=np.diff(L_tofit)
        
    #deleting the discrepancies
    thr=np.max(abs(diff))*0.05
    idx_diff= np.where(abs(diff)>thr)[0]+1
        
    #new slim down of data
    L_tofit2=np.delete(L_tofit, idx_diff)
    T_tofit2=np.delete(T_tofit, idx_diff)
        
    #check for enough points
    if len(L_tofit2) < 30:
        L_tofit2=L_tofit
        T_tofit2=T_tofit
        
    L_fit=L_tofit2
    T_fit=T_tofit2   
    
    L_fit=np.array(L_fit)
    T_fit=np.array(T_fit)
      
    #normalization of the fit interval    
    norm_T_fit=[]
    norm_T_fit=np.array(norm_T_fit)
    for element in T_fit:
        z=(element-np.amin(T_fit))/(np.amax(T_fit)-np.amin(T_fit))
        norm_T_fit=np.append(norm_T_fit, z)
      
    #defining the fit function
    def fit(x, a, b, c, d):
        return (a*np.exp((-b)*x))+(c*np.exp((-d)*x))

    model=Model(fit)    

    #performing fit of last segments of data
    model.set_param_hint('b', value=0.2, min=0, max=100)
    model.set_param_hint('d', value=0.2, min=0, max=100)
    fit_result=model.fit(L_fit, x=norm_T_fit, a=1, b=0.2, c=1, d=0.2)
    
    #transforming the times from unix in seconds
    T_fit_real=T_fit-np.amin(T_fit)  
    index=np.where(T_fit_real<=1800)
    #evaluating the fit luminosity    
    Y=fit(T_fit_real, fit_result.params['a'].value, (fit_result.params['b'].value/(np.amax(T_fit)-np.amin(T_fit))), fit_result.params['c'].value, fit_result.params['d'].value/(np.amax(T_fit)-np.amin(T_fit)))
    with open('Data/L30m16.txt', 'a') as f:
        f.write(str(Y[index][-1]))
        f.write('\n')
   
   
#2017  
with open('Data/L30m17.txt', 'w') as f:
  f.write('')
  f.close()

for i in range(len(FillNumber17)):
    #plotting results 
    plt.close("all")
    text = str(int(FillNumber17[i])) #number of current fill
    #obtain the Times and luminosity evolution values for that fill
    f=open('ATLAS/ATLAS_fill_2017/{}_lumi_ATLAS.txt'.format(text),"r")
    lines=f.readlines()
    L_evolx=[]
    times=[]
    for x in lines:
        times.append(int(x.split(' ')[0]))  
        L_evolx.append(float(x.split(' ')[2]))
          
    f.close()
    Times = np.array(times)
    L_evol = np.array(L_evolx)
    
    
    #deleting the null values of the luminosity
    zero=np.where(L_evol<100)
    L_zero=np.delete(L_evol, zero)
    T_zero=np.delete(Times, zero)
        
    #check for enough points
    if len(L_zero)<10:
        zero=np.where(L_evol<5)
        L_zero=np.delete(L_evol, zero)
        T_zero=np.delete(Times, zero)

    #defining the derivative 
    dy = np.zeros(L_zero.shape)
    dy[0:-1] = np.diff(L_zero)/np.diff(T_zero)

     #start to slim down the fit interval       
    L_tofit=[]
    T_tofit=[]
    for idx in range(len(L_zero)):
        #cancelling too strong derivative points
        if dy[idx]<0 and dy[idx]>-1.5:
            L_tofit.append(L_zero[idx])
            T_tofit.append(T_zero[idx])
        if dy[idx]>0 or dy[idx]<-1.5:
            continue     
        
    #evaluating the differences between two subsequent points
    diff=np.diff(L_tofit)
        
    #deleting the discrepancies
    thr=np.max(abs(diff))*0.05
    idx_diff= np.where(abs(diff)>thr)[0]+1
        
    #new slim down of data
    L_tofit2=np.delete(L_tofit, idx_diff)
    T_tofit2=np.delete(T_tofit, idx_diff)
        
    #check for enough points
    if len(L_tofit2) < 30:
        L_tofit2=L_tofit
        T_tofit2=T_tofit
        
    L_fit=L_tofit2
    T_fit=T_tofit2   
    
    L_fit=np.array(L_fit)
    T_fit=np.array(T_fit)
      
    #normalization of the fit interval    
    norm_T_fit=[]
    norm_T_fit=np.array(norm_T_fit)
    for element in T_fit:
        z=(element-np.amin(T_fit))/(np.amax(T_fit)-np.amin(T_fit))
        norm_T_fit=np.append(norm_T_fit, z)
      
    #defining the fit function
    def fit(x, a, b, c, d):
        return (a*np.exp((-b)*x))+(c*np.exp((-d)*x))

    model=Model(fit)    

    #performing fit of last segments of data
    model.set_param_hint('b', value=0.2, min=0, max=100)
    model.set_param_hint('d', value=0.2, min=0, max=100)
    fit_result=model.fit(L_fit, x=norm_T_fit, a=1, b=0.2, c=1, d=0.2)
    
    #transforming the times from unix in seconds
    T_fit_real=T_fit-np.amin(T_fit)  
    index=np.where(T_fit_real<=1800)
    #evaluating the fit luminosity    
    Y=fit(T_fit_real, fit_result.params['a'].value, (fit_result.params['b'].value/(np.amax(T_fit)-np.amin(T_fit))), fit_result.params['c'].value, fit_result.params['d'].value/(np.amax(T_fit)-np.amin(T_fit)))
    with open('Data/L30m17.txt', 'a') as f:
        f.write(str(Y[index][-1]))
        f.write('\n')     
        
        
#2018       
with open('Data/L30m18.txt', 'w') as f:
  f.write('')
  f.close()

for i in range(len(FillNumber18)):
    #plotting results 
    plt.close("all")
    text = str(int(FillNumber18[i])) #number of current fill
    #obtain the Times and luminosity evolution values for that fill
    f=open('ATLAS/ATLAS_fill_2018/{}_lumi_ATLAS.txt'.format(text),"r")
    lines=f.readlines()
    L_evolx=[]
    times=[]
    for x in lines:
        times.append(int(x.split(' ')[0]))  
        L_evolx.append(float(x.split(' ')[2]))
          
    f.close()
    Times = np.array(times)
    L_evol = np.array(L_evolx)
    
    
    #deleting the null values of the luminosity
    zero=np.where(L_evol<100)
    L_zero=np.delete(L_evol, zero)
    T_zero=np.delete(Times, zero)
        
    #check for enough points
    if len(L_zero)<10:
        zero=np.where(L_evol<5)
        L_zero=np.delete(L_evol, zero)
        T_zero=np.delete(Times, zero)

    #defining the derivative 
    dy = np.zeros(L_zero.shape)
    dy[0:-1] = np.diff(L_zero)/np.diff(T_zero)

     #start to slim down the fit interval       
    L_tofit=[]
    T_tofit=[]
    for idx in range(len(L_zero)):
        #cancelling too strong derivative points
        if dy[idx]<0 and dy[idx]>-1.5:
            L_tofit.append(L_zero[idx])
            T_tofit.append(T_zero[idx])
        if dy[idx]>0 or dy[idx]<-1.5:
            continue     
        
    #evaluating the differences between two subsequent points
    diff=np.diff(L_tofit)
        
    #deleting the discrepancies
    thr=np.max(abs(diff))*0.05
    idx_diff= np.where(abs(diff)>thr)[0]+1
        
    #new slim down of data
    L_tofit2=np.delete(L_tofit, idx_diff)
    T_tofit2=np.delete(T_tofit, idx_diff)
        
    #check for enough points
    if len(L_tofit2) < 30:
        L_tofit2=L_tofit
        T_tofit2=T_tofit
        
    L_fit=L_tofit2
    T_fit=T_tofit2   
    
    L_fit=np.array(L_fit)
    T_fit=np.array(T_fit)
      
    #normalization of the fit interval    
    norm_T_fit=[]
    norm_T_fit=np.array(norm_T_fit)
    for element in T_fit:
        z=(element-np.amin(T_fit))/(np.amax(T_fit)-np.amin(T_fit))
        norm_T_fit=np.append(norm_T_fit, z)
      
    #defining the fit function
    def fit(x, a, b, c, d):
        return (a*np.exp((-b)*x))+(c*np.exp((-d)*x))

    model=Model(fit)    

    #performing fit of last segments of data
    model.set_param_hint('b', value=0.2, min=0, max=100)
    model.set_param_hint('d', value=0.2, min=0, max=100)
    fit_result=model.fit(L_fit, x=norm_T_fit, a=1, b=0.2, c=1, d=0.2)
    
    #transforming the times from unix in seconds
    T_fit_real=T_fit-np.amin(T_fit)  
    index=np.where(T_fit_real<=1800)
    #evaluating the fit luminosity    
    Y=fit(T_fit_real, fit_result.params['a'].value, (fit_result.params['b'].value/(np.amax(T_fit)-np.amin(T_fit))), fit_result.params['c'].value, fit_result.params['d'].value/(np.amax(T_fit)-np.amin(T_fit)))
    with open('Data/L30m18.txt', 'a') as f:
        f.write(str(Y[index][-1]))
        f.write('\n')