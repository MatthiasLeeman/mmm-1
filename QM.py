import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import *
from PIL import Image, ImageDraw
from io import BytesIO

video= 'False'
video_filename= 'D:\School\Master2\MMM\output_video.gif'
Number_of_timesteps= int(1e5) 
discr_order= "fourth" #choose fourth or second order accurate discretization of Laplacian operator


Ly= 50*10**(-9) #m
dy= 0.125*10**(-9) #m
kvalues= np.arange(0, Ly//dy+1)
dt= 4.1696*10**(-18) #s

tc= Number_of_timesteps/4* dt
sigma_t= tc/6


e_mass= 9.109383*10**(-31) #kg
omega_HO= 50*10**(12) #rad/s
q= -elementary_charge

def create_plot(probdens):
    y = kvalues*dy
    plt.plot(y, probdens)
    plt.xlabel('y')
    plt.ylabel('Probability Density')
    plt.grid(True)
    # Save the plot to memory buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

def potential(y):
    return e_mass* omega_HO**2 * (y-Ly/2)**2 /2

def ground_state(y):
    return ((e_mass*omega_HO)/(np.pi*hbar))**(1/4) * np.exp(-(e_mass*omega_HO / (2*hbar)) * (y-Ly/2)**2)

def gaussian_pulse(t, amplitude= 1e8):
    return amplitude * np.exp(-(t-tc)**2 / (2*sigma_t**2))

PsiR= np.zeros(int(Ly//dy)+1)
PsiI= np.zeros(int(Ly//dy)+1)


prob_dens= np.zeros((int(Ly//dy)+1, Number_of_timesteps))
expectation_pos= np.zeros(Number_of_timesteps)
for k in range(len(PsiR)):
    PsiR[k]= ground_state(k*dy)
    PsiR[0]= 0
    PsiR[-1]= 0
Psi= PsiR + 1j* PsiI

prob_dens[:,0]= np.real(np.conjugate(Psi) * Psi)
expectation_pos[0]= np.sum(dy * kvalues * prob_dens[:,0]*dy) #prob density times dy gives prob so we get y*P(y)



# Generate movie from frames
frame_rate = 100  # Number of frames per second
frame_width, frame_height = 640, 480  # Size of frames

output_frames = []

for n in range(1,Number_of_timesteps):
    match discr_order:
        case "second":
            PsiR[1:-1]= PsiR[1:-1] - hbar*dt/(2*e_mass) * (PsiI[2:] - 2* PsiI[1:-1] + PsiI[:-2]) / dy**2 - dt/hbar * (q*(kvalues[1:-1]*dy - Ly/2)* gaussian_pulse(n*dt) - potential(kvalues[1:-1]*dy)) * PsiI[1:-1]
            PsiI[1:-1]= PsiI[1:-1] + hbar*dt/(2*e_mass) * (PsiR[2:] - 2* PsiR[1:-1] + PsiR[:-2]) / dy**2 + dt/hbar * (q*(kvalues[1:-1]*dy - Ly/2) * gaussian_pulse(n*dt) - potential(kvalues[1:-1]*dy)) * PsiR[1:-1]
            
        case "fourth":
            PsiR[2:-2]= PsiR[2:-2] - hbar*dt/(2*e_mass) * (-PsiI[4:] + 16*PsiI[3:-1] - 30*PsiI[2:-2] + 16*PsiI[1:-3] - PsiI[:-4]) / (12*dy**2) - dt/hbar * (q*(kvalues[2:-2]*dy - Ly/2) * gaussian_pulse(n*dt) - potential(kvalues[2:-2]*dy)) * PsiI[2:-2]
            PsiI[2:-2]= PsiI[2:-2] + hbar*dt/(2*e_mass) * (-PsiR[4:] + 16*PsiR[3:-1] - 30*PsiR[2:-2] + 16*PsiR[1:-3] - PsiR[:-4]) / (12*dy**2) + dt/hbar * (q*(kvalues[2:-2]*dy - Ly/2) * gaussian_pulse(n*dt) - potential(kvalues[2:-2]*dy))* PsiR[2:-2]
    
    Psi= PsiR + 1j* PsiI
    prob_dens[:,n]= np.real(np.conjugate(Psi) * Psi)
    expectation_pos[n]= np.sum(dy * kvalues * prob_dens[:,n]*dy)

    if video == 'True':
        print("video")
        buf = create_plot(prob_dens[:,n])
        
        # Convert plot to frame
        img = Image.open(buf)
        img = img.resize((frame_width, frame_height), Image.ANTIALIAS)
        output_frames.append(img)

# Save frames as video
if video == 'True':
    output_frames[0].save(video_filename, save_all=True, append_images=output_frames[1:], loop=0, duration= Number_of_timesteps*dt// frame_rate)
# Create some mock data

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('Timestep')
ax1.set_ylabel('Gaussian Pulse', color=color)
ax1.plot(np.arange(Number_of_timesteps), gaussian_pulse(np.arange(Number_of_timesteps)*dt), color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Expectation value of position', color=color)  # we already handled the x-label with ax1
ax2.plot(np.arange(Number_of_timesteps), expectation_pos, color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, Ly)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

