import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.io import loadmat
from sys import argv

#points = loadmat("pi_coordinates.mat")['z'].T[0]
points = np.array([1, 1j]).dot(np.loadtxt(open(argv[1], "rb"), delimiter=",").T)
points = (points.real + 1j * points.imag)
plt.show()
N = len(points)
T = np.linspace(0,2*pi,1000)

# Fourier Transform and Rearrangement of data
f_coef = np.fft.fft(points)/N
f_freq = np.fft.fftfreq(len(points))*N

f_coef = f_coef[-np.argsort(abs(f_freq))]
f_freq = f_freq[-np.argsort(abs(f_freq))]

# Coefficient Adder
def coef_adder(n,t):
    sum = 0
    for i in range(n+1): sum += f_coef[i] * np.exp(1j*f_freq[i]*t)
    return sum

# Animation of Pi
fig = plt.figure()
ax = plt.axes(xlim=(-300,300),ylim=(-300,300))
ax.set_aspect("equal")
circle = [plt.Circle((0,0),radius=0.0,fill=False,color="black") for _ in range(N)]
line = [plt.plot([],[],color="black")[0] for _ in range(N)]
curve = curve = plt.plot([],[],color="red")[0]
xdata,ydata = [],[]
for cir in circle: ax.add_artist(cir)

def animate(t):
    for i in range(N-1):
        x1, y1 = coef_adder(i,t).real, coef_adder(i,t).imag
        x2, y2 = coef_adder(i+1, t).real, coef_adder(i+1, t).imag

        circle[i].center = coef_adder(i,t).real, coef_adder(i,t).imag
        circle[i].radius = abs(f_coef[i])
        line[i].set_data([x1,x2],[y1,y2])

    xdata.append(coef_adder(N-1,t).real)
    ydata.append(coef_adder(N-1,t).imag)
    curve.set_data(xdata,ydata)
    return [curve] + circle + line

anim = FuncAnimation(fig,animate,frames=T,interval=15,repeat=True,blit=True)
#anim.save("Pi.mp4",writer="ffmpeg")
plt.show()
