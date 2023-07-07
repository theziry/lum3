import numpy as np
import matplotlib.pyplot as plt
import LoadData as ld


plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Helvetica",
  "font.size": 12
})

#loading fill number
FillNumber16, FillNumber17, FillNumber18 = ld.FillNumber()


#loading data
#luminosity value at 30 minutes - 2017
f=open('Data/L30m17.txt',"r")
lines=f.readlines()
L30m17=[]
for x in lines:
    L30m17.append(float(x.split(' ')[0]))  

f.close()

#luminosity value at 30 minutes - 2018
f=open('Data/L30m18.txt',"r")
lines=f.readlines()
L30m18=[]
for x in lines:
    L30m18.append(float(x.split(' ')[0]))  

f.close()

L30m17=np.array(L30m17)

#finding the mode of the 2017 sample
counts, bins = np.histogram(L30m17)
max_bin = np.argmax(counts)
mode=bins[max_bin:max_bin+2].mean()

#finding the mode of the first 40 fills of 2018
firsts=L30m18[:40]
counts1, bins1 = np.histogram(firsts)
max_bin1 = np.argmax(counts1)
mode1=bins1[max_bin1:max_bin1+2].mean()

#plotting the data distribution
fig, ax1=plt.subplots()
n,bins, c = ax1.hist(L30m17, density=True)
ax1.axvline(mode, color='r')
ax1.set_xlabel('Luminosity [Hz/$\mu$b]')
ax1.set_ylabel('Normalized Frequency')
ax1.set_title('30 min Luminosity Distribution 2017')
plt.show()

fig, ax1=plt.subplots()
n,bins, c = ax1.hist(firsts, density=True)
ax1.axvline(mode1, color='r')
ax1.set_xlabel('Luminosity [Hz/$\mu$b]')
ax1.set_ylabel('Normalized Frequency')
ax1.set_title('30 min Luminosity Distribution - first values of 2018')
plt.show()

count=[]
for i in range(len(L30m18)):
  if L30m18[i]<mode:
    count.append(0)
    print(FillNumber18[i])
    #print("False")
  elif L30m18[i]>mode:
    count.append(1)
    print(FillNumber18[i])
    #print("True")

#loading data
f=open('Data/res_opt_2018.txt',"r")
lines=f.readlines()
opt_18=[]
for x in lines:
    opt_18.append(float(x.split(' ')[0]))  

f.close()


count2=[]
for i in range(len(opt_18)):
  if opt_18[i]<mode:
    count2.append(0)
  elif opt_18[i]>mode:
    count2.append(1)
    
count=np.array(count)
count2=np.array(count2)
check=count2-count
print("Online Strategy")
print(count)
print("Posterior Strategy")
print(count2)

print("Differences")
print(check)
print(np.where(check==1)[0])
print(len(np.where(check==1)[0]), "over", len(check))


fig2, ax2=plt.subplots()
ax2.plot(FillNumber18, count, "r+", label='Online Optimization')
ax2.plot(FillNumber18, count2, "bx", label='A Posteriori Optimization')
ax2.plot([],[], label='Differences={}/{}'.format(len(np.where(check==1)[0]), len(check)))
ax2.set_xlabel('Fill Number')
ax2.set_ylabel('Fill Extension')
ax2.set_title('Online Optimization with 2017 fills')
plt.legend(loc="best")
plt.show()




count3=[]
for i in range(40):
  if L30m18[i]<mode:
    count3.append(0)
    #print("False")
  elif L30m18[i]>mode:
    count3.append(1)
    #print("True")

for i in range(40, len(L30m18)):
  if L30m18[i]<mode1:
    count3.append(0)
  elif L30m18[i]>mode1:
    count3.append(1)

count3=np.array(count3)
check2=count2-count3
print("Online Strategy")
print(count3)
print("Posterior Strategy")
print(count2)

print("Differences")
print(check2)
print(np.where(check2==1)[0])
print(len(np.where(check2==1)[0]), "over", len(check2))


fig2, ax2=plt.subplots()
ax2.plot(FillNumber18, count3, "r+", label='Online Optimization')
ax2.plot(FillNumber18, count2, "bx", label='A Posteriori Optimization')
ax2.plot([],[], label='Differences={}/{}'.format(len(np.where(check2==1)[0]), len(check2)))
ax2.set_xlabel('Fill Number')
ax2.set_ylabel('Fill Extension')
ax2.set_title('Online Optimization with 2018 fills')
plt.legend(loc="best")
plt.show()