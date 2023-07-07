from lmfit import Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad

#Importing datas from the excel file
data = pd.read_excel(r'eof_prob.xlsx')

df1 = pd.DataFrame(data, columns=['p2016']).dropna()
df2 = pd.DataFrame(data, columns=['p2017']).dropna()
df3 = pd.DataFrame(data, columns=['p2018']).dropna()
dfb16= pd.DataFrame(data, columns=['Bins16']).dropna()
dfb17= pd.DataFrame(data, columns=['Bins17']).dropna()
dfb18= pd.DataFrame(data, columns=['Bins18']).dropna()

#Converting data into array
p16 = np.array(df1['p2016'].values.tolist())
b16= np.array(dfb16['Bins16'].values.tolist())
p17 = np.array(df2['p2017'].values.tolist())
b17= np.array(dfb17['Bins17'].values.tolist())
p18 = np.array(df3['p2018'].values.tolist())
b18= np.array(dfb18['Bins18'].values.tolist())

#defining the fit function
def fit(x, a, b, c, d):
    return (a*np.exp((-b)*x))+(c*np.exp((-d)*x))

model=Model(fit) 

#performing the fit
fit_result=model.fit(p16, x=b16, a=1, b=0.3, c=1, d=0.2)
y=fit(b16, fit_result.params['a'].value, fit_result.params['b'].value, fit_result.params['c'].value, fit_result.params['d'].value)

#normalization of data 
b16_norm=(b16-np.amin(b16))/(np.amax(b16)-np.amin(b16))
print(b16, b16_norm)
fit_result_norm=model.fit(p16, x=b16_norm, a=1, b=0.1, c=1, d=0.2)
y_norm=fit(b16_norm, fit_result_norm.params['a'].value, fit_result_norm.params['b'].value, fit_result_norm.params['c'].value, fit_result_norm.params['d'].value)

#Expectation value evaluation
fun1=lambda x: x*((fit_result_norm.params['a'].value*np.exp((-fit_result_norm.params['b'].value)*x))+((fit_result_norm.params['c'].value)*np.exp((-fit_result_norm.params['d'].value)*x)))
int= quad(fun1, 0, 1)[0]


#finding the mode 
counts, bins = np.histogram(b16)
max_bin = np.argmax(counts)
mode=bins[max_bin:max_bin+2].mean()


#plotting results
fig1, ax1=plt.subplots()
ax1.axvline(np.average(b16), color='k',linestyle='dashed',linewidth=0.75,label='Average Value={:0.2f} [h]'.format(np.average(b16)) )
ax1.axvline((int*(np.amax(b16)-np.amin(b16)))+np.amin(b16),linestyle='dashed', linewidth=0.75,color='g', label='Expectation Value={:0.2f} [h]'.format((int*(np.amax(b16)-np.amin(b16)))+np.amin(b16)))
ax1.axvline(mode, color='orange', linestyle='dashed',linewidth=0.75,label='Mode={:0.2f} [h]'.format((mode)))
ya=fit(np.average(b16), fit_result.params['a'].value, fit_result.params['b'].value, fit_result.params['c'].value, fit_result.params['d'].value)
ax1.axhline(ya, color='k', linestyle='dashed',linewidth=0.75,label='p Average Value={:0.2f}'.format(ya) )
yb=fit((int*(np.amax(b16)-np.amin(b16)))+np.amin(b16), fit_result.params['a'].value, fit_result.params['b'].value, fit_result.params['c'].value, fit_result.params['d'].value)
ax1.axhline(yb, color='g', linestyle='dashed',linewidth=0.75,label='p Expectation Value={:0.2f}'.format(yb))
yc=fit(mode, fit_result.params['a'].value, fit_result.params['b'].value, fit_result.params['c'].value, fit_result.params['d'].value)
ax1.axhline(yc, color='orange',linestyle='dashed', linewidth=0.75, label='p Mode={:0.2f}'.format((yc)))
ax1.plot(b16, y, 'r-', label='Best Fit')
ax1.plot(b16, p16, "bx", label='Data')
ax1.set_xlabel('Bins [h]')
ax1.set_ylabel('Probability of reaching the end of fill')
plt.legend(loc='best')
plt.show()

