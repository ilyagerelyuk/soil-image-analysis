import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from math import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import max_error

w=0.7 #параметр контрольных допусков. Меньше - больше лакун и наоборот
testnums=[5,8,9,10,12]
dataset=3
wb=False

if wb:
    for i in range(1,15):
        for t in range(2,5):
            newsize=900
            imgname="photos/"+str(i)+"-"+str(t)+".jpg"
            print(imgname)
            im = cv2.imread(imgname)
            xsize, ysize, zsize = im.shape
            centerx = xsize//2+50
            centery = ysize//2-100
            R=im[:,:,2].astype(np.int32)
            G=im[:,:,1].astype(np.int32)
            B=im[:,:,0].astype(np.int32)
            ones=np.full(
                  shape=R.shape,
                  fill_value=255,
                  dtype=np.int32
                )
            D=np.sqrt((ones-R)**2+(ones-G)**2+(ones-B)**2)
            print(R[0,0], G[0,0], B[0,0], D[0,0])
            Rbest=np.mean(R[D<200])
            Gbest=np.mean(G[D<200])
            Bbest=np.mean(B[D<200])
            print(Rbest, Gbest, Bbest)
            k1=255/Rbest
            k2=255/Gbest
            k3=255/Bbest
            print(k1,k2,k3)
            R=np.minimum((R*k1).astype(int),ones)
            G=np.minimum((G*k2).astype(int),ones)
            B=np.minimum((B*k3).astype(int),ones)
            output=np.stack([B,G,R], axis=2)
            output = output[centerx-newsize//2:centerx+newsize//2+newsize%2, centery-newsize//2:centery+newsize//2+newsize%2,:]
            cv2.imwrite('images/'+str(i)+"-"+str(t)+'.jpg', output)


X=[]
Y=[]

with open("info.txt") as file:
    for line in file:
        s=line.rstrip()
        num, gumus = s.split()
        num=int(num)
        gumus=float(gumus)
        X.append(num)
        Y.append(gumus)
        
plotx=[]
ploty=[]
testx=[]
testy=[]
for t in [dataset]:
    qw=0
    for n in X:
        #print(n, "-", t, " +")
        im = cv2.imread("images/"+str(n)+"-"+str(t)+".jpg")
        size = im.shape[0]
        numpix = size*size
        im = im.astype(np.float32)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        H=im[:,:,0]
        S=im[:,:,1]
        V=im[:,:,2]
        #Поиск мод
        H_hist=np.histogram(H, bins=360, range=(0,360))[0]
        S_hist=np.histogram(S, bins=100, range=(0,1))[0]
        V_hist=np.histogram(V, bins=100, range=(0,1))[0]
        meanV=np.mean(V)
        absV=np.abs(V-meanV)
        mdV=np.mean(absV) #Средний модуль отклонения яркости
        #mdV=meanV #Убрать
        modeH=np.argmax(H_hist)
        modeS=np.argmax(S_hist)/100
        modeV=np.argmax(V_hist)/100
        medianV=np.median(V)
        mask=np.where(V<medianV+mdV*w, 1, 0)*np.where(V>medianV-mdV*w, 1, 0)*np.where(H>modeH-15, 1, 0)*np.where(H<modeH+15, 1, 0)
        
        H=H[mask==1]
        S=S[mask==1]
        V=V[mask==1]
        meanH=np.mean(H)
        medianH=np.median(H)
        medianS=np.median(S)
        H_hist=np.histogram(H, bins=360, range=(0,360))[0]
        S_hist=np.histogram(S, bins=100, range=(0,1))[0]
        V_hist=np.histogram(V, bins=100, range=(0,1))[0]
        modeH=np.argmax(H_hist)
        modeS=np.argmax(S_hist)/100
        modeV=np.argmax(V_hist)/100
        H_hist0=H_hist[H_hist>0]
        varHp=np.var(H_hist0)

        f=np.fft.fft(H_hist) #Спектральная фильтрация
        f=np.fft.fftshift(f)
        s1=179
        s2=180
        for i in range(360):
            if min(abs(i-s1), abs(i-s2))>30:
                f[i]=0
        inv=np.fft.ifftshift(f)
        inv=np.abs(np.real(np.fft.ifft(f)))
        inv[:max(modeH-15, 0)]=0
        inv[min(360,modeH+16):]=0
        H_hist0=inv[inv>0]
        varH02=np.var(H_hist0)
        modef=np.argmax(inv)
        
        res=[modeH, varHp, modef, varH02, modeS]
        print(res)
        if (n in testnums):
            testx.append(res)
            testy.append(Y[qw])
        else:
            plotx.append(res)
            ploty.append(Y[qw])
        qw+=1

X_np = np.array(plotx)
Y_np = np.array(ploty)
X_test = np.array(testx)
Y_test = np.array(testy)
sc=StandardScaler()
X_np = sc.fit_transform(X_np)
X_test = sc.transform(X_test)

regr = Ridge(alpha=1.0)

regr.fit(X_np,Y_np)

y_pred=regr.predict(X_np)
y_test_pred=regr.predict(X_test)
plotx=list(y_pred.flatten())
plotxtest=list(y_test_pred)

plt.scatter(plotx, ploty, label="Train")
plt.scatter(plotxtest, testy, label="Test", marker="^")
train_mae = mean_absolute_error(ploty, plotx)
train_max = max_error(ploty, plotx)
test_mae = mean_absolute_error(testy, plotxtest)
test_max = max_error(testy, plotxtest)
print("Train:")
print("MAE =", train_mae)
print("MAX =", train_max)
print("TEST:")
print("MAE =", test_mae)
print("MAX =", test_max)
plt.plot([1.2, 3.4], [1.2, 3.4], color='k')
plt.xlabel("Гумус, предсказанное содержание, %")
plt.ylabel("Лабораторный анализ. Гумус, %")
plt.legend()
plt.show()
