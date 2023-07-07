import numpy as np
import matplotlib.pyplot as plt
import LoadData as ld
import LuminosityOptimization as lo



#Evaluating the double exponential data model considering the correct data sample

#loading data    
FillNumber16, FillNumber17, FillNumber18 = ld.FillNumber()   

L_int_summary_16=[]
for i in range(len(FillNumber16)):   
    text = str(int(FillNumber16[i])) #number of current fill
    #obtain the Times and luminosity evolution values for that fill 
    f=open('ATLAS/ATLAS_summary_2016/{}_summary_ATLAS.txt'.format(text),"r")
    lines=f.readlines()
    for x in lines: 
        L_int_summary_16.append(float(x.split(' ')[3]))
          
    f.close()

L_int_summary_16 = np.array(L_int_summary_16)



L_int_summary_17=[]
for i in range(len(FillNumber17)):   
    text = str(int(FillNumber17[i])) #number of current fill
    #obtain the Times and luminosity evolution values for that fill 
    f=open('ATLAS/ATLAS_summary_2017/{}_summary_ATLAS.txt'.format(text),"r")
    lines=f.readlines()
    for x in lines: 
        L_int_summary_17.append(float(x.split(' ')[3]))
          
    f.close()

L_int_summary_17 = np.array(L_int_summary_17)


L_int_summary_18=[]
for i in range(len(FillNumber18)):   
    text = str(int(FillNumber18[i])) #number of current fill
    #obtain the Times and luminosity evolution values for that fill 
    f=open('ATLAS/ATLAS_summary_2018/{}_summary_ATLAS.txt'.format(text),"r")
    lines=f.readlines()
    for x in lines: 
        L_int_summary_18.append(float(x.split(' ')[3]))
          
    f.close()

L_int_summary_18 = np.array(L_int_summary_18)
