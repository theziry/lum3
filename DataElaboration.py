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

#Evaluating the double exponential data model considering the correct data sample

#loading data
data_16, data_17, data_18, array16, array17, array18 = ld.Data()
data_tot, dataTot, arrayTot = ld.TotalDataSet(data_16, data_17, data_18)
data_ta16, data_tf16, data_ta17, data_tf17, data_ta18, data_tf18 = ld.loadFill()     
FillNumber16, FillNumber17, FillNumber18 = ld.FillNumber()
data_16_sec, data_ta16_sec, data_tf16_sec, data_17_sec, data_ta17_sec,\
   data_tf17_sec, data_18_sec, data_ta18_sec, data_tf18_sec=ld.Data_sec(array16,\
      data_ta16, data_tf16, array17, data_ta17, data_tf17, array18, data_ta18, data_tf18)
   


#2016
L_intfit16=[]

a_16=[]
b_16=[]
b1_16=[]
d1_16=[]
c_16=[]
d_16=[]
a1_16=[]
c1_16=[]

norm_max_16=[]
norm_min_16=[]
ts_16=[]
tn_16=[]
reduced_chi4_16=[]
L_int_2016=[]

for i in range(len(FillNumber16)):
    #plotting results 
    plt.close("all")
    fig1,  ax1 = plt.subplots()
    print("######################################################################FILL",int(FillNumber16[i]),"#########################################################")
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
    L_i1=integrate.simps(L_fit, T_fit) 
    print("Luminosity of the unix time=", L_i1)       
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
    #print(fit_result.params['a'].value, fit_result.params['b'].value/(np.amax(T_fit)-np.amin(T_fit)), fit_result.params['c'].value, fit_result.params['d'].value/(np.amax(T_fit)-np.amin(T_fit)))
    ax1.plot(T_fit, L_fit, "b.", label='Smoothed data', markersize=4)
    
    #transforming the times from unix in seconds
    T_fit_real=T_fit-np.amin(T_fit)  
    #saving values for numerical optimization 
    ts_16.append(T_fit_real[len(T_fit_real)-1])
    tn_16.append(norm_T_fit[len(norm_T_fit)-1])
    norm_min_16.append(np.amin(T_fit))
    norm_max_16.append(np.amax(T_fit))
    
    #evaluating the fit luminosity    
    Y=fit(T_fit_real, fit_result.params['a'].value, (fit_result.params['b'].value/(np.amax(T_fit)-np.amin(T_fit))), fit_result.params['c'].value, fit_result.params['d'].value/(np.amax(T_fit)-np.amin(T_fit)))
    #defining the plots
    ax1.plot(T_fit, Y, 'r-', label='Double exponential fit')
    ax1.plot([], [], 'kx ', label=r'$\tilde{\chi}^2$='+'{:.5f}'.format(fit_result.redchi))
    ax1.set_xlabel('Times [s]')
    ax1.set_ylabel('Luminosity evolution [$\mathrm{Hz}/\mathrm{\mu b}$]')
    plt.legend(loc='best')
    
    #evaluating luminosity
    #np.amin(T_fit))
    Y1=fit(T_fit_real, fit_result.params['a'].value, (fit_result.params['b'].value/(np.amax(T_fit)-np.amin(T_fit))), fit_result.params['c'].value, fit_result.params['d'].value/(np.amax(T_fit)-np.amin(T_fit)))
    L_i=integrate.simps(Y1,T_fit_real)
    print("My model luminosity=", L_i)
    y2=fit(T_fit_real, fit_result.params['a'].value/(np.amax(T_fit)-np.amin(T_fit)), (fit_result.params['b'].value/(np.amax(T_fit)-np.amin(T_fit))), fit_result.params['c'].value/(np.amax(T_fit)-np.amin(T_fit)), fit_result.params['d'].value/(np.amax(T_fit)-np.amin(T_fit)))
    L_i2=integrate.simps(y2,T_fit_real)
    print("a_mod luminosity=", L_i2)
    y3=fit(norm_T_fit, fit_result.params['a'].value, fit_result.params['b'].value, fit_result.params['c'].value, fit_result.params['d'].value)
    L_i3=integrate.simps(y3,norm_T_fit)
    L_int_2016.append(L_i)
    print("Normalized luminosity=", L_i3)
    print("L_i2*K=", L_i2*(np.amax(T_fit)-np.amin(T_fit)))
    L_intfit16.append(L_i)
    
    #saving fit parameters
    a_16.append(fit_result.params['a'].value)
    b_16.append(fit_result.params['b'].value/(np.amax(T_fit)-np.amin(T_fit)))
    c_16.append(fit_result.params['c'].value)
    d_16.append(fit_result.params['d'].value/(np.amax(T_fit)-np.amin(T_fit)))
    b1_16.append(fit_result.params['b'].value)
    d1_16.append(fit_result.params['d'].value)
    a1_16.append(fit_result.params['a'].value/(np.amax(T_fit)-np.amin(T_fit)))
    c1_16.append(fit_result.params['c'].value/(np.amax(T_fit)-np.amin(T_fit))) 
    ax1.set_title('Fill {}'.format(text)) 
    
    #saving the red_chi values
    reduced_chi4_16.append(fit_result.redchi)
    
    #saving the figure
    plt.savefig('FitModel/{}_fitModel.pdf'.format(text)) 
       

#ploting the parameter distributions
plt.close('all')
fig3, ax3=plt.subplots(figsize=(8, 5))
ax3.hist(a_16,  color="green", alpha=0.3, density=True)
ax3.set_title('a - First amplitude 2016')
ax3.set_ylabel('Normalized Frequencies')
ax3.set_xlabel('Parameter Values')
plt.savefig('FitModel/a_2016.pdf') 
plt.show()
plt.close()
fig3, ax3=plt.subplots()
ax3.hist(b_16, color="green", alpha=0.3, density=True)
ax3.set_title('b - First decay constant 2016')
ax3.set_ylabel('Normalized Frequencies')
ax3.set_xlabel('Parameter Values')
plt.savefig('FitModel/b_2016.pdf') 
plt.show()
plt.close()
fig3, ax3=plt.subplots(figsize=(8, 5))
ax3.hist(c_16, color="green", alpha=0.3, density=True)
ax3.set_title('c - Second amplitude 2016')
ax3.set_ylabel('Normalized Frequencies')
ax3.set_xlabel('Parameter Values')
plt.savefig('FitModel/c_2016.pdf')
plt.show()
plt.close()
fig3, ax3=plt.subplots()
ax3.hist(d_16, color="green", alpha=0.3, density=True)
ax3.set_title('d - Second decay constant 2016')
ax3.set_ylabel('Normalized Frequencies')
ax3.set_xlabel('Parameter Values')
plt.savefig('FitModel/d_2016.pdf') 
plt.show()
plt.close()


#2017
L_intfit17=[]

a_17=[]
b_17=[]
c_17=[]
d_17=[]
b1_17=[]
d1_17=[]
a1_17=[]
c1_17=[]

norm_min_17=[]
norm_max_17=[]
ts_17=[]
tn_17=[]
reduced_chi4_17=[]
L_int_2017=[]
for i in range(len(FillNumber17)):
    #plotting results 
    plt.close("all")
    fig1,  ax1 = plt.subplots()
    print("######################################################################FILL",int(FillNumber17[i]),"#########################################################")
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
    print(fit_result.params['a'].value, fit_result.params['b'].value/(np.amax(T_fit)-np.amin(T_fit)), fit_result.params['c'].value, fit_result.params['d'].value/(np.amax(T_fit)-np.amin(T_fit)), fit_result.params['b'].value, fit_result.params['d'].value )
    ax1.plot(T_fit, L_fit, "b.", label='Smoothed data', markersize=4)
    
    #transforming the times from unix in seconds
    T_fit_real=T_fit-np.amin(T_fit)  
    #saving values for numerical optimization 
    ts_17.append(T_fit_real[len(T_fit_real)-1])
    tn_17.append(norm_T_fit[len(norm_T_fit)-1])
    norm_min_17.append(np.amin(T_fit))
    norm_max_17.append(np.amax(T_fit))
    
    #evaluating the fit luminosity    
    Y=fit(T_fit_real, fit_result.params['a'].value, (fit_result.params['b'].value/(np.amax(T_fit)-np.amin(T_fit))), fit_result.params['c'].value, fit_result.params['d'].value/(np.amax(T_fit)-np.amin(T_fit)))
    
    #defining the plots
    ax1.plot(T_fit, Y, 'r-', label='Double exponential fit')
    ax1.plot([], [], 'kx ', label=r'$\tilde{\chi}^2$='+'{:.5f}'.format(fit_result.redchi))
    ax1.set_xlabel('Times [s]')
    ax1.set_ylabel('Luminosity evolution [$\mathrm{Hz}/\mathrm{\mu b}$]')
    plt.legend(loc='best')
        
    #evaluating luminosity
    L_i=integrate.simps(fit_result.best_fit, T_fit)
    L_intfit17.append(L_i)
    y2=fit(T_fit_real, fit_result.params['a'].value, (fit_result.params['b'].value/(np.amax(T_fit)-np.amin(T_fit))), fit_result.params['c'].value, fit_result.params['d'].value/(np.amax(T_fit)-np.amin(T_fit)))
    L_i2=integrate.simps(y2,T_fit_real)
    L_int_2017.append(L_i2)
    
    #saving fit parameters
    a_17.append(fit_result.params['a'].value)
    b_17.append(fit_result.params['b'].value/(np.amax(T_fit)-np.amin(T_fit)))
    c_17.append(fit_result.params['c'].value)
    d_17.append(fit_result.params['d'].value/(np.amax(T_fit)-np.amin(T_fit)))
    b1_17.append(fit_result.params['b'].value)
    d1_17.append(fit_result.params['d'].value) 
    a1_17.append(fit_result.params['a'].value/(np.amax(T_fit)-np.amin(T_fit)))
    c1_17.append(fit_result.params['c'].value/(np.amax(T_fit)-np.amin(T_fit)))   
    ax1.set_title('Fill {}'.format(text)) 
    
    #saving the red_chi values
    reduced_chi4_17.append(fit_result.redchi)
    
    #saving the figure
    plt.savefig('FitModel/{}_fitModel.pdf'.format(text))  

#ploting the parameter distributions
plt.close('all')
fig3, ax3=plt.subplots()
n3, bins3, patches3 = ax3.hist(a_17, color="steelblue", alpha=0.5, density=True)
ax3.set_title('a - First amplitude 2017')
ax3.set_ylabel('Normalized Frequencies')
ax3.set_xlabel('Parameter Values')
plt.savefig('FitModel/a_2017.pdf')
##plt.show()
plt.close()
fig3, ax3=plt.subplots()
ax3.hist(b_17, color="steelblue", alpha=0.5, density=True)
ax3.set_title('b - First decay constant 2017')
ax3.set_ylabel('Normalized Frequencies')
ax3.set_xlabel('Parameter Values')
plt.savefig('FitModel/b_2017.pdf')
##plt.show()
plt.close()
fig3, ax3=plt.subplots()
ax3.hist(c_17, color="steelblue", alpha=0.5, density=True)
ax3.set_title('c - Second amplitude 2017')
ax3.set_ylabel('Normalized Frequencies')
ax3.set_xlabel('Parameter Values')
plt.savefig('FitModel/c_2017.pdf') 
##plt.show()
plt.close()
fig3, ax3=plt.subplots()
ax3.hist(d_17, color="steelblue", alpha=0.5, density=True)
ax3.set_title('d - Second decay constant 2017')
ax3.set_ylabel('Normalized Frequencies')
ax3.set_xlabel('Parameter Values')
plt.savefig('FitModel/d_2017.pdf') 
##plt.show()
plt.close()

#2018
L_intfit18=[]

a_18=[]
b_18=[]
c_18=[]
d_18=[]
b1_18=[]
d1_18=[]
a1_18=[]
c1_18=[]

norm_max_18=[]
norm_min_18=[]
ts_18=[]
tn_18=[]
reduced_chi4_18=[]
L_int_2018=[]
for i in range(len(FillNumber18)):
    #plotting results 
    plt.close("all")
    fig1,  ax1 = plt.subplots()
    print("######################################################################FILL",int(FillNumber18[i]),"#########################################################")
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
    print(fit_result.params['a'].value, fit_result.params['b'].value/(np.amax(T_fit)-np.amin(T_fit)), fit_result.params['c'].value, fit_result.params['d'].value/(np.amax(T_fit)-np.amin(T_fit)))
    ax1.plot(T_fit, L_fit, "b.", label='Smoothed data', markersize=4)
    
    #transforming the times from unix in seconds
    T_fit_real=T_fit-np.amin(T_fit)  
    
    #saving values for numerical optimization 
    ts_18.append(T_fit_real[len(T_fit_real)-1])
    norm_min_18.append(np.amin(T_fit))
    norm_max_18.append(np.amax(T_fit)) 
    tn_18.append(tn_18)
           
    #evaluating the fit luminosity    
    Y=fit(T_fit_real, fit_result.params['a'].value, (fit_result.params['b'].value/(np.amax(T_fit)-np.amin(T_fit))), fit_result.params['c'].value, fit_result.params['d'].value/(np.amax(T_fit)-np.amin(T_fit)))
    
    #defining the plots
    ax1.plot(T_fit, Y, 'r-', label='Double exponential fit')
    ax1.plot([], [], 'kx ', label=r'$\tilde{\chi}^2$ ='+'{:.5f}'.format(fit_result.redchi))
    ax1.set_xlabel('Times [s]')
    ax1.set_ylabel('Luminosity evolution [$\mathrm{Hz}/\mathrm{\mu b}$]')
    plt.legend(loc='best')
    
    #evaluating luminosity
    L_i=integrate.simps(fit_result.best_fit, T_fit)
    L_intfit18.append(L_i)
    y2=fit(T_fit_real, fit_result.params['a'].value, (fit_result.params['b'].value/(np.amax(T_fit)-np.amin(T_fit))), fit_result.params['c'].value, fit_result.params['d'].value/(np.amax(T_fit)-np.amin(T_fit)))
    L_i2=integrate.simps(y2,T_fit_real)
    L_int_2018.append(L_i2)
          
    #saving fit parameters
    a_18.append(fit_result.params['a'].value)
    b_18.append(fit_result.params['b'].value/(np.amax(T_fit)-np.amin(T_fit)))
    c_18.append(fit_result.params['c'].value)
    d_18.append(fit_result.params['d'].value/(np.amax(T_fit)-np.amin(T_fit)))
    b1_18.append(fit_result.params['b'].value)
    d1_18.append(fit_result.params['d'].value)
    a1_18.append(fit_result.params['a'].value/(np.amax(T_fit)-np.amin(T_fit)))
    c1_18.append(fit_result.params['c'].value/(np.amax(T_fit)-np.amin(T_fit))) 
    ax1.set_title('Fill {}'.format(text)) 
    
    #saving the red_chi values
    reduced_chi4_18.append(fit_result.redchi)
    
    #saving the figure
    plt.savefig('FitModel/{}_fitModel.pdf'.format(text)) 
    ###plt.show()     

#ploting the parameter distributions
plt.close('all')
fig3, ax3=plt.subplots()
ax3.hist(a_18, color="pink", alpha=0.8, density=True)
ax3.set_title('a - First amplitude 2018 ')
ax3.set_ylabel('Normalized Frequencies')
ax3.set_xlabel('Parameter Values')
plt.savefig('FitModel/a_2018.pdf') 
##plt.show()
plt.close()
fig3, ax3=plt.subplots()
ax3.hist(b_18,color="pink", alpha=0.8, density=True)
ax3.set_title('b - First decay constant 2018')
ax3.set_ylabel('Normalized Frequencies')
ax3.set_xlabel('Parameter Values')
plt.savefig('FitModel/b_2018.pdf') 
##plt.show()
plt.close()
fig3, ax3=plt.subplots()
ax3.hist(c_18, color="pink", alpha=0.8,density=True)
ax3.set_title('c - Second amplitude 2018')
ax3.set_ylabel('Normalized Frequencies')
ax3.set_xlabel('Parameter Values')
plt.savefig('FitModel/c_2018.pdf') 
###plt.show()
plt.close()
fig3, ax3=plt.subplots()
ax3.hist(d_18, color="pink", alpha=0.8, density=True)
ax3.set_title('d - Second decay constant 2018')
ax3.set_ylabel('Normalized Frequencies')
ax3.set_xlabel('Parameter Values')
plt.savefig('FitModel/d_2018.pdf') 
###plt.show()
plt.close()

  
"""
#Correlation between parameters
corr1=np.corrcoef(a_16, b_16)
corr2=np.corrcoef(c_16, d_16)
corr3=np.corrcoef(a_16, c_16)
corr4=np.corrcoef(a_16, d_16)
corr5=np.corrcoef(b_16, d_16)
corr6=np.corrcoef(c_16, b_16)

print(corr1[0,1], corr2[0,1], corr3[0,1], corr4[0,1], corr5[0,1], corr6[0,1])

#Correlation Plots
plt.close('all')
fig4, ax4=plt.subplots()
ax4.plot(a_16, b_16, "b.")
ax4.set_title('a16/b16')
ax4.set_ylabel('b_16')
ax4.set_xlabel('a_16')
ax4.plot([],[],"k>", label="Correlation={:.3f}".format(corr1[0,1]))
plt.legend(loc="best")
plt.savefig('FitModel/a16_b16.pdf') 
plt.close()
fig4, ax4=plt.subplots()
ax4.plot(a_16, d_16, "b.")
ax4.set_title('a16/d16')
ax4.set_ylabel('d_16')
ax4.set_xlabel('a_16')
ax4.plot([],[],"k>", label="Correlation={:.3f}".format(corr4[0,1]))
plt.legend(loc="best")
plt.savefig('FitModel/a16_d16.pdf') 
plt.close()
fig4, ax4=plt.subplots()
ax4.plot(b_16, d_16, "b.")
ax4.set_title('b16/d16')
ax4.set_ylabel('d_16')
ax4.set_xlabel('b_16')
ax4.plot([],[],"k>", label="Correlation={:.3f}".format(corr5[0,1]))
plt.legend(loc="best")
plt.savefig('FitModel/b16_d16.pdf') 
plt.close()
fig4, ax4=plt.subplots()
ax4.plot(c_16, b_16, "b.")
ax4.set_title('c16/b16')
ax4.set_ylabel('b_16')
ax4.set_xlabel('c_16')
ax4.plot([],[],"k>", label="Correlation={:.3f}".format(corr6[0,1]))
plt.legend(loc="best")
plt.savefig('FitModel/c16_b16.pdf') 
plt.close()
fig4, ax4=plt.subplots()
ax4.plot(a_16, c_16, "b.")
ax4.set_title('a16/c16')
ax4.set_ylabel('c_16')
ax4.set_xlabel('a_16')
ax4.plot([],[],"k>", label="Correlation={:.3f}".format(corr3[0,1]))
plt.legend(loc="best")
plt.savefig('FitModel/a16_c16.pdf') 
plt.close()
fig4, ax4=plt.subplots()
ax4.plot(c_16, d_16, "b.")
ax4.set_title('c16/d16')
ax4.set_ylabel('d_16')
ax4.set_xlabel('c_16')
ax4.plot([],[],"k>", label="Correlation={:.3f}".format(corr2[0,1]))
plt.legend(loc="best")
plt.savefig('FitModel/c16_d16.pdf') 
plt.close()

corr1=np.corrcoef(a_17, b_17)
corr2=np.corrcoef(c_17, d_17)
corr3=np.corrcoef(a_17, c_17)
corr4=np.corrcoef(a_17, d_17)
corr5=np.corrcoef(b_17, d_17)
corr6=np.corrcoef(c_17, b_17)

print(corr1[0,1], corr2[0,1], corr3[0,1], corr4[0,1], corr5[0,1], corr6[0,1])

fig4, ax4=plt.subplots()
ax4.plot(a_17, b_17, "b.")
ax4.set_title('a17/b17')
ax4.set_ylabel('$b_17$')
ax4.set_xlabel('$a_17$')
ax4.plot([],[],"k>", label="Correlation={:.3f}".format(corr1[0,1]))
plt.legend(loc="best")
plt.savefig('FitModel/a17_b17.pdf') 
plt.close()
fig4, ax4=plt.subplots()
ax4.plot(a_17, d_17, "b.")
ax4.set_title('a17/d17')
ax4.set_ylabel('$d_17$')
ax4.set_xlabel('$a_17$')
ax4.plot([],[],"k>", label="Correlation={:.3f}".format(corr4[0,1]))
plt.legend(loc="best")
plt.savefig('FitModel/a17_d17.pdf') 
plt.close()
fig4, ax4=plt.subplots()
ax4.plot(b_17, d_17, "b.")
ax4.set_title('b17/d17')
ax4.set_ylabel('$d_17$')
ax4.set_xlabel('$b_17$')
ax4.plot([],[],"k>", label="Correlation={:.3f}".format(corr5[0,1]))
plt.legend(loc="best")
plt.savefig('FitModel/b17_d17.pdf') 
plt.close()
fig4, ax4=plt.subplots()
ax4.plot(a_17, c_17, "b.")
ax4.set_title('a17/c17')
ax4.set_ylabel('$c_17$')
ax4.set_xlabel('$a_17$')
ax4.plot([],[],"k>", label="Correlation={:.3f}".format(corr6[0,1]))
plt.legend(loc="best")
plt.savefig('FitModel/a17_c17.pdf') 
plt.close()
fig4, ax4=plt.subplots()
ax4.plot(c_17, d_17, "b.")
ax4.set_title('c17/d17')
ax4.set_ylabel('$d_17$')
ax4.set_xlabel('$c_17$')
ax4.plot([],[],"k>", label="Correlation={:.3f}".format(corr3[0,1]))
plt.legend(loc="best")
plt.savefig('FitModel/c17_d17.pdf') 
plt.close()
fig4, ax4=plt.subplots()
ax4.plot(b_17, c_17, "b.")
ax4.set_title('b17/c17')
ax4.set_ylabel('$c_17$')
ax4.set_xlabel('$b_17$')
ax4.plot([],[],"k>", label="Correlation={:.3f}".format(corr2[0,1]))
plt.legend(loc="best")
plt.savefig('FitModel/b17_c17.pdf') 
plt.close()

corr1=np.corrcoef(a_18, b_18)
corr2=np.corrcoef(c_18, d_18)
corr3=np.corrcoef(a_18, c_18)
corr4=np.corrcoef(a_18, d_18)
corr5=np.corrcoef(b_18, d_18)
corr6=np.corrcoef(c_18, b_18)

print(corr1[0,1], corr2[0,1], corr3[0,1], corr4[0,1], corr5[0,1], corr6[0,1])

fig4, ax4=plt.subplots()
ax4.plot(a_18, b_18, "b.")
ax4.set_title('a18/b18')
ax4.set_ylabel('b_18')
ax4.set_xlabel('a_18')
ax4.plot([],[],"k>", label="Correlation={:.3f}".format(corr1[0,1]))
plt.legend(loc="best")
plt.savefig('FitModel/a18_b18.pdf') 
plt.close()
fig4, ax4=plt.subplots()
ax4.plot(a_18, d_18, "b.")
ax4.set_title('a18/d18')
ax4.set_ylabel('d_18')
ax4.set_xlabel('a_18')
ax4.plot([],[],"k>", label="Correlation={:.3f}".format(corr4[0,1]))
plt.legend(loc="best")
plt.savefig('FitModel/a18_d18.pdf') 
plt.close()
fig4, ax4=plt.subplots()
ax4.plot(b_18, d_18, "b.")
ax4.set_title('b18/d18')
ax4.set_ylabel('d_18')
ax4.set_xlabel('b_18')
ax4.plot([],[],"k>", label="Correlation={:.3f}".format(corr5[0,1]))
plt.legend(loc="best")
plt.savefig('FitModel/b18_d18.pdf') 
plt.close()
fig4, ax4=plt.subplots()
ax4.plot(a_18, c_18, "b.")
ax4.set_title('a18/c18')
ax4.set_ylabel('c_18')
ax4.set_xlabel('a_18')
ax4.plot([],[],"k>", label="Correlation={:.3f}".format(corr6[0,1]))
plt.legend(loc="best")
plt.savefig('FitModel/a18_c18.pdf') 
plt.close()
fig4, ax4=plt.subplots()
ax4.plot(c_18, d_18, "b.")
ax4.set_title('c18/d18')
ax4.set_ylabel('d_18')
ax4.set_xlabel('c_18')
ax4.plot([],[],"k>", label="Correlation={:.3f}".format(corr3[0,1]))
plt.legend(loc="best")
plt.savefig('FitModel/c18_d18.pdf') 
plt.close()
fig4, ax4=plt.subplots()
ax4.plot(b_18, c_18, "b.")
ax4.set_title('b18/c18')
ax4.set_ylabel('c_18')
ax4.set_xlabel('b_18')
ax4.plot([],[],"k>", label="Correlation={:.3f}".format(corr2[0,1]))
plt.legend(loc="best")
plt.savefig('FitModel/b18_c18.pdf') 
plt.close()
"""

fig1, ax1=plt.subplots()
ax1.plot(FillNumber16, reduced_chi4_16, "b.", label=r'4 Parameters $\tilde{\chi}^2$')
ax1.set_title("2016")
plt.legend(loc="best")
plt.savefig('Redchi/4Par_2016_scatter.pdf')
#plt.show()
fig1, ax1=plt.subplots()
ax1.plot(FillNumber17, reduced_chi4_17, "b.", label=r'4 Parameters $\tilde{\chi}^2$')
ax1.set_title("2017")
plt.legend(loc="best")
plt.savefig('Redchi/4Par_2017_scatter.pdf')
#plt.show()
fig1, ax1=plt.subplots()
ax1.plot(FillNumber18, reduced_chi4_18, "b.", label=r'4 Parameters $\tilde{\chi}^2$')
ax1.set_title("2018")
plt.legend(loc="best")
plt.savefig('Redchi/4Par_2018_scatter.pdf')
#plt.show()

fig1, ax1=plt.subplots()
ax1.hist(reduced_chi4_16,  color="green", alpha=0.3, label=r'4 Parameters $\tilde{\chi}^2$')
ax1.set_title("2016")
plt.legend(loc="best")
ax1.set_ylabel('Normalized Frequencies')
ax1.set_xlabel('Reduced Chi Square')
plt.savefig('Redchi/4Par_2016_distr.pdf')
#plt.show()
fig1, ax1=plt.subplots()
ax1.hist(reduced_chi4_17, color="steelblue", alpha=0.5,label=r'4 Parameters $\tilde{\chi}^2$')
ax1.set_title("2017")
plt.legend(loc="best")
ax1.set_ylabel('Normalized Frequencies')
ax1.set_xlabel('Reduced Chi Square')
plt.savefig('Redchi/4Par_2017_distr.pdf')
#plt.show()
fig1, ax1=plt.subplots()
ax1.hist(reduced_chi4_18, color="pink", alpha=0.8, label=r'4 Parameters $\tilde{\chi}^2$')
ax1.set_title("2018")
plt.legend(loc="best")
ax1.set_ylabel('Normalized Frequencies')
ax1.set_xlabel('Reduced Chi Square')
plt.savefig('Redchi/4Par_2018_distr.pdf')
#plt.show()



with open('Data/a_16_4Par.txt', 'w') as f:
        f.write('')
        f.close()
for el in a_16:
    with open('Data/a_16_4Par.txt', 'a') as f:
        f.write(str(el))
        f.write('\n')
        
with open('Data/b_16_4Par.txt', 'w') as f:
        f.write('')
        f.close()
for el in b_16:
    with open('Data/b_16_4Par.txt', 'a') as f:
        f.write(str(el))
        f.write('\n')
        
with open('Data/c_16_4Par.txt', 'w') as f:
        f.write('')
        f.close()
for el in c_16:
    with open('Data/c_16_4Par.txt', 'a') as f:
        f.write(str(el))
        f.write('\n')
        
with open('Data/d_16_4Par.txt', 'w') as f:
        f.write('')
        f.close()
for el in d_16:
    with open('Data/d_16_4Par.txt', 'a') as f:
        f.write(str(el))
        f.write('\n')
        
with open('Data/a_17_4Par.txt', 'w') as f:
        f.write('')
        f.close()
for el in a_17:
    with open('Data/a_17_4Par.txt', 'a') as f:
        f.write(str(el))
        f.write('\n')
        
with open('Data/b_17_4Par.txt', 'w') as f:
        f.write('')
        f.close()
for el in b_17:
    with open('Data/b_17_4Par.txt', 'a') as f:
        f.write(str(el))
        f.write('\n')
        
with open('Data/c_17_4Par.txt', 'w') as f:
        f.write('')
        f.close()
for el in c_17:
    with open('Data/c_17_4Par.txt', 'a') as f:
        f.write(str(el))
        f.write('\n')
        
with open('Data/d_17_4Par.txt', 'w') as f:
        f.write('')
        f.close()
for el in d_17:
    with open('Data/d_17_4Par.txt', 'a') as f:
        f.write(str(el))
        f.write('\n')
        
with open('Data/a_18_4Par.txt', 'w') as f:
        f.write('')
        f.close()
for el in a_18:
    with open('Data/a_18_4Par.txt', 'a') as f:
        f.write(str(el))
        f.write('\n')
        
with open('Data/b_18_4Par.txt', 'w') as f:
        f.write('')
        f.close()
for el in b_18:
    with open('Data/b_18_4Par.txt', 'a') as f:
        f.write(str(el))
        f.write('\n')
        
with open('Data/c_18_4Par.txt', 'w') as f:
        f.write('')
        f.close()
for el in c_18:
    with open('Data/c_18_4Par.txt', 'a') as f:
        f.write(str(el))
        f.write('\n')
        
with open('Data/d_18_4Par.txt', 'w') as f:
        f.write('')
        f.close()
for el in d_18:
    with open('Data/d_18_4Par.txt', 'a') as f:
        f.write(str(el))
        f.write('\n')
        
        
with open('Data/ts_16_4Par.txt', 'w') as f:
        f.write('')
        f.close()
for el in ts_16:
    with open('Data/ts_16_4Par.txt', 'a') as f:
        f.write(str(el))
        f.write('\n')
        
with open('Data/ts_17_4Par.txt', 'w') as f:
        f.write('')
        f.close()
for el in ts_17:
    with open('Data/ts_17_4Par.txt', 'a') as f:
        f.write(str(el))
        f.write('\n')
        
with open('Data/ts_18_4Par.txt', 'w') as f:
        f.write('')
        f.close()
for el in ts_18:
    with open('Data/ts_18_4Par.txt', 'a') as f:
        f.write(str(el))
        f.write('\n')
        
with open('Data/L_int_2016_4Par.txt', 'w') as f:
        f.write('')
        f.close()
for el in L_int_2016:
    with open('Data/L_int_2016_4Par.txt', 'a') as f:
        f.write(str(el))
        f.write('\n')

with open('Data/L_int_2017_4Par.txt', 'w') as f:
        f.write('')
        f.close()
for el in L_int_2017:
    with open('Data/L_int_2017_4Par.txt', 'a') as f:
        f.write(str(el))
        f.write('\n')

with open('Data/L_int_2018_4Par.txt', 'w') as f:
        f.write('')
        f.close()
for el in L_int_2018:
    with open('Data/L_int_2018_4Par.txt', 'a') as f:
        f.write(str(el))
        f.write('\n')