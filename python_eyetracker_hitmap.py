import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import matplotlib.cm as cm
import pandas as pd


np.random.seed(0)
xrange = 1920
yrange = 1080
n = 100
x = np.random.randint(0, 1920, n)
y = np.random.randint(0, 1080, n)

data = pd.read_csv('sample_picture.csv')

x_pos = data['x_position'].values
y_pos = data['y_position'].values
data_time = data['time'].values

for i in range(len(x_pos)):
    print(" x_pos data = ", end='')
    print(type(x_pos[i]))

    print("y_pos data  = ", end='')
    print(type(y_pos[i]))


img = np.zeros([yrange,xrange])

for i in range(len(x_pos)):
    img[int(y_pos[i])][int(x_pos[i])] = img[int(y_pos[i])][int(x_pos[i])] + 1

img2 = ndi.gaussian_filter(img, 16)

#############################################################

ex2 = np.zeros((1,0), dtype=int)
ex2 = np.append(ex2, 0)

for i in range(1,len(x_pos)):
    v = np.sqrt((x_pos[i] - x_pos[i-1])**2 + (y_pos[i] - y_pos[i-1])**2)
    vt = v/data_time[i]

    # 1 = fixation
    # 2 = saccade

    if vt <200 :
        ex2 = np.append(ex2, 1)
    elif vt >=200:
        ex2 = np.append(ex2, 2)
    else:
        ex2 = np.append(ex2, 0)

    # print(vt)

print(ex2)

v_bool = False

ex3 = np.zeros((1,0), dtype=int)
tmp_start = 0
tmp_end = 0

for i in range(len(ex2)):

    if ex2[i] ==1 and v_bool == True:
        pass
    elif ex2[i] == 1:
        v_bool = True
        tmp_start = i

    elif ex2[i] == 2 and v_bool == True:
        v_bool = False
        tmp_end = i - 1
        real_end = tmp_end
        while ex2[real_end] != 1:
            real_end = real_end -1

        print('tmp_start = ', end='')
        print(tmp_start)

        print('tmp_end = ', end='')
        print(tmp_end)
        print('real_end = ', end='')
        print(real_end)

        print('')

        fixation_center = int((tmp_start + tmp_end)/2)
        ex3 = np.append(ex3, fixation_center)

print(ex3)
count = 1
for i in range(len(ex3)):
    plt.scatter(x_pos[ex3[i]], y_pos[ex3[i]])
    plt.text(x_pos[ex3[i]], y_pos[ex3[i]], count)
    count = count + 1


##############################################################

for i in range(len(ex3)):
    plt.scatter(x_pos[ex3[i]], y_pos[ex3[i]])
    plt.plot(x_pos[ex3[i]], y_pos[ex3[i]])
# plt.plot(x_pos,y_pos,'k.')
plt.imshow(img2, cmap=cm.jet)
plt.show()

