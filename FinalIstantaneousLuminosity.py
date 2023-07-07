import numpy as np
import matplotlib.pyplot as plt
import LoadData as ld
import scipy.integrate as integrate
from lmfit import Model
import matplotlib.ticker as mticker
from RealIntegratedLuminosity import L_int_summary_16, L_int_summary_17, L_int_summary_18

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
   
with open('EndLumi/SavedFills16.txt', 'w') as f:
    f.write('')
    f.close()
    
with open('EndLumi/IgnoredFills16.txt', 'w') as f:
    f.write('')
    f.close()

with open('EndLumi/SavedFills17.txt', 'w') as f:
    f.write('')
    f.close()
    
with open('EndLumi/IgnoredFills17.txt', 'w') as f:
    f.write('')
    f.close()

with open('EndLumi/SavedFills18.txt', 'w') as f:
    f.write('')
    f.close()
    
with open('EndLumi/IgnoredFills18.txt', 'w') as f:
    f.write('')
    f.close()

#2016
L_intfit16=[]
L_int_2016=[]
L_ist_tend_16=[]

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
    
    #evaluating the fit luminosity    
    Y=fit(T_fit_real, fit_result.params['a'].value, (fit_result.params['b'].value/(np.amax(T_fit)-np.amin(T_fit))), fit_result.params['c'].value, fit_result.params['d'].value/(np.amax(T_fit)-np.amin(T_fit)))
    #defining the plots
    ax1.plot(T_fit, Y, 'r-', label='Double exponential fit')
    ax1.plot([], [], 'kx ', label=r'$\tilde{\chi}^2$='+'{:.5f}'.format(fit_result.redchi))
    ax1.set_xlabel('Times [s]')
    ax1.set_ylabel('Luminosity evolution [$\mathrm{Hz}/\mathrm{\mu b}$]')
    plt.legend(loc='best')
    
    #evaluating integrated luminosity and verifying the goodness of the model --> searching for the ending values of L_ist
    L_i=integrate.simps(Y,T_fit_real)
    print("My model luminosity=", L_i)
    L_intfit16.append(L_i)
    if abs(L_i-L_int_summary_16[i])<=(0.008*L_int_summary_16[i]): #setting as the error for the extrapolation a 0.8% of difference
        print('Good Extrapolation!')
        L_ist_tend_16.append(Y[-1])
        with open('EndLumi/SavedFills16.txt', 'a') as f:
                f.write(text)
                f.write(' ')
                f.write(str(Y[-1]))
                f.write('\n')
    elif abs(L_i-L_int_summary_16[i])>(0.008*L_int_summary_16[i]):
        with open('EndLumi/IgnoredFills16.txt', 'a') as f:
            f.write(text)
            f.write('\n')
    ax1.set_title('Fill {}'.format(text)) 
    
    
    #saving the figure
    plt.savefig('FitModel/{}_fitModel.pdf'.format(text)) 

plt.close('all')      
fig2, ax2=plt.subplots()
ax2.hist(L_ist_tend_16, density=True, facecolor='green', alpha=0.4, label='2016')
ax2.set_xlabel('Istantaneous Luminosity [$\mathrm{Hz}/\mathrm{\mu b}$]')
ax2.set_ylabel('Normalized Frequencies')
ax2.set_title('Istantaneous Luminosity of the end of fill')
plt.legend(loc='best')
plt.savefig('EndLumi/EndLumi_2016.pdf')



with open('EndLumi/IgnoredFills16.txt', 'a') as f:
    f.write('Number of ignored fills: ')
    f.write(str(len(L_intfit16)-len(L_ist_tend_16)))
    f.write('\n')
    
with open('EndLumi/SavedFills16.txt', 'a') as f:
    f.write('Number of saved fills: ')
    f.write(str(len(L_ist_tend_16)))
    f.write('\n')    
    
#2017
L_ist_tend_17=[]
L_intfit17=[]
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
    
    #evaluating the fit luminosity    
    Y=fit(T_fit_real, fit_result.params['a'].value, (fit_result.params['b'].value/(np.amax(T_fit)-np.amin(T_fit))), fit_result.params['c'].value, fit_result.params['d'].value/(np.amax(T_fit)-np.amin(T_fit)))
    
    #defining the plots
    ax1.plot(T_fit, Y, 'r-', label='Double exponential fit')
    ax1.plot([], [], 'kx ', label=r'$\tilde{\chi}^2$='+'{:.5f}'.format(fit_result.redchi))
    ax1.set_xlabel('Times [s]')
    ax1.set_ylabel('Luminosity evolution [$\mathrm{Hz}/\mathrm{\mu b}$]')
    plt.legend(loc='best')
        
    #evaluating integrated luminosity and verifying the goodness of the model --> searching for the ending values of L_ist
    L_i=integrate.simps(Y,T_fit_real)
    print("My model luminosity=", L_i)
    L_intfit17.append(L_i)
    if abs(L_i-L_int_summary_17[i])<=(0.008*L_int_summary_17[i]): #setting as the error for the extrapolation a 0.8% of difference
        print('Good Extrapolation!')
        L_ist_tend_17.append(Y[-1])
        with open('EndLumi/SavedFills17.txt', 'a') as f:
                f.write(text)
                f.write(' ')
                f.write(str(Y[-1]))
                f.write('\n')
    elif abs(L_i-L_int_summary_17[i])>(0.008*L_int_summary_17[i]):
        with open('EndLumi/IgnoredFills17.txt', 'a') as f:
            f.write(text)
            f.write('\n')
    ax1.set_title('Fill {}'.format(text)) 
    
    #saving the figure
    plt.savefig('FitModel/{}_fitModel.pdf'.format(text)) 
    
plt.close('all')   
  
fig2, ax2=plt.subplots()
ax2.hist(L_ist_tend_17, density=True, facecolor='steelblue', label='2017')
ax2.set_xlabel('Istantaneous Luminosity [$\mathrm{Hz}/\mathrm{\mu b}$]')
ax2.set_ylabel('Normalized Frequencies')
ax2.set_title('Istantaneous Luminosity of the End of Fill')
plt.legend(loc='best')
plt.savefig('EndLumi/EndLumi_2017.pdf')

with open('EndLumi/IgnoredFills17.txt', 'a') as f:
    f.write('Number of ignored fills: ')
    f.write(str(len(L_intfit17)-len(L_ist_tend_17)))
    f.write('\n')
    
with open('EndLumi/SavedFills17.txt', 'a') as f:
    f.write('Number of saved fills: ')
    f.write(str(len(L_ist_tend_17)))
    f.write('\n')    

#2018
L_ist_tend_18=[]
L_intfit18=[]
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
           
    #evaluating the fit luminosity    
    Y=fit(T_fit_real, fit_result.params['a'].value, (fit_result.params['b'].value/(np.amax(T_fit)-np.amin(T_fit))), fit_result.params['c'].value, fit_result.params['d'].value/(np.amax(T_fit)-np.amin(T_fit)))
    
    #defining the plots
    ax1.plot(T_fit, Y, 'r-', label='Double exponential fit')
    ax1.plot([], [], 'kx ', label=r'$\tilde{\chi}^2$ ='+'{:.5f}'.format(fit_result.redchi))
    ax1.set_xlabel('Times [s]')
    ax1.set_ylabel('Luminosity evolution [$\mathrm{Hz}/\mathrm{\mu b}$]')
    plt.legend(loc='best')
    
    #evaluating integrated luminosity and verifying the goodness of the model --> searching for the ending values of L_ist
    L_i=integrate.simps(Y,T_fit_real)
    print("My model luminosity=", L_i)
    L_intfit18.append(L_i)
    if abs(L_i-L_int_summary_18[i])<=(0.008*L_int_summary_18[i]): #setting as the error for the extrapolation a 0.8% of difference
        print('Good Extrapolation!')
        L_ist_tend_18.append(Y[-1])
        with open('EndLumi/SavedFills18.txt', 'a') as f:
                f.write(text)
                f.write(' ')
                f.write(str(Y[-1]))
                f.write('\n')
    elif abs(L_i-L_int_summary_18[i])>(0.008*L_int_summary_18[i]):
        with open('EndLumi/IgnoredFills18.txt', 'a') as f:
            f.write(text)
            f.write('\n')
    ax1.set_title('Fill {}'.format(text))
    
    #saving the figure
    plt.savefig('FitModel/{}_fitModel.pdf'.format(text)) 
    
plt.close('all')   
  
fig2, ax2=plt.subplots()
ax2.hist(L_ist_tend_18, density=True, facecolor='pink', label='2018')
ax2.set_xlabel('Istantaneous Luminosity [$\mathrm{Hz}/\mathrm{\mu b}$]')
ax2.set_ylabel('Normalized Frequencies')
ax2.set_title('Istantaneous Luminosity of the End of Fill')
plt.legend(loc='best')
plt.savefig('EndLumi/EndLumi_2018.pdf')

with open('EndLumi/IgnoredFills18.txt', 'a') as f:
    f.write('Number of ignored fills: ')
    f.write(str(len(L_intfit18)-len(L_ist_tend_18)))
    f.write('\n')
    
with open('EndLumi/SavedFills18.txt', 'a') as f:
    f.write('Number of saved fills: ')
    f.write(str(len(L_ist_tend_18)))
    f.write('\n')     

