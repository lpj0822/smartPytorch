from visdom import Visdom
import numpy as np
import matplotlib.pyplot as plt

viz = Visdom(env='my_wind')
x,y,z=0,0,0
win = viz.line(
    X=np.array([x]),
    Y=np.column_stack((np.array([y]),np.array([z]))),
    opts=dict(title='two_lines', color="red"))

for i in range(10):
    y+=i
    z+=2
    viz.line(
        X=np.column_stack((np.array([x]), np.array([1]))),
        Y=np.column_stack((np.array([y]), np.array([z]))),
        win=win,
        update='append')

a = [10, 15]
b = [3, 4]
#viz.bar(X=np.array(a), Y=np.array(b))

fig=plt.figure()
plt.plot(range(1,10), range(1,10), 'b*')
plt.show()

viz.matplot(fig)