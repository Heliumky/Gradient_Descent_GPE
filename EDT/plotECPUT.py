import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import plotsetting as ps

# Load data (example data)
gd20sd13 = np.loadtxt("GD2_CPUTIME_seed13.txt")
gd20sd14 = np.loadtxt("GD2_CPUTIME_seed14.txt")
gd20sd15 = np.loadtxt("GD2_CPUTIME_seed15.txt")
gd20 = np.loadtxt("GD2_CPUTIME_D20.txt")
gd30 = np.loadtxt("GD2_CPUTIME_D30.txt")
gd40 = np.loadtxt("GD2_CPUTIME_D40.txt")
tdvp20 = np.loadtxt("TDVP_CPUTIME_D20.txt")
tdvp30 = np.loadtxt("TDVP_CPUTIME_D30.txt")
tdvp40 = np.loadtxt("TDVP_CPUTIME_D40.txt")



Fort = np.loadtxt("FOR_CPUTIME.txt")
YK_imag = np.loadtxt("YK_imag_mu.txt")
#Exact_E = 4.354341506267
Exact_E = 5.759742281197
E_dt = 5.759742281197
Trotter = np.abs(Exact_E - E_dt)
# Create figure and axis with adjusted size
fig, ax = plt.subplots()

#ax.set_ylim(1e-13, 1e2)
# Ensure axes autoscale
ax.relim()
ax.autoscale_view()

# set title
ax.set_title('Pure GP CPU TIME')
#print(gd30[:,0])
# Scatter plot for Intel Fortran and one-site TDVP
ax.plot(gd20sd13[:,0], np.abs(gd20sd13[:,1]-Exact_E), label="GD20sd13", color='black')
ax.plot(gd20sd14[:,0], np.abs(gd20sd14[:,1]-Exact_E), label="GD20sd14", color='red', linestyle='-.')
ax.plot(gd20sd15[:,0], np.abs(gd20sd15[:,1]-Exact_E), label="GD20sd15", color='green', linestyle='--')
#ax.plot(tdvp20[:,0], tdvp20[:,1]-Exact_E, label="TDVPD20", color='red')
#ax.plot(tdvp30[:,0], tdvp30[:,1]-Exact_E, label="TDVPD30", color='red', linestyle='-.')
#ax.plot(tdvp40[:,0], np.abs(tdvp40[:,1]-Exact_E), label="TDVPD40", color='red', linestyle='--')
#ax.plot(Fort[:,0], np.abs(Fort[:,1]-Exact_E), label="Tranditional", color='green')

#ax.plot(np.linspace(0,553.222530*5,len(YK_imag)), np.abs(YK_imag-Exact_E), label="Tranditional-YK-IM", color='blue')
#ax.plot(range(len(YK_imag)), [Trotter]*len(YK_imag), label="Trotter Err", color='blue', linestyle='--')
# Linear fit to one-site TDVP data
#coefficients = np.polyfit(tdvp[:,0][3:], tdvp[:,1][3:], deg=1)
#poly = np.poly1d(coefficients)
#tdvp_fit = np.linspace(min(tdvp[:,0]), max(tdvp[:,0]), 100)
#ax.plot(tdvp_fit, poly(tdvp_fit), color='red', linewidth=2, linestyle='--')

# Linear fit to Foftran data
#coefficients_ft = np.polyfit(intelf[:,0], np.log(intelf[:,1]), deg=1)
#poly_ft = np.poly1d(coefficients_ft)
#ft_fit = np.linspace(min(intelf[:,0]), 12, 100)  # Adjusted range for better visibility
#ax.plot(ft_fit, np.exp(poly_ft(ft_fit)), color='black', linewidth=2, linestyle='--')

# Set labels and title
ax.set_xlabel(r'CPU time[s]')
ax.set_ylabel(r'$E_{err}$')
ax.set_yscale('log')
#ax.set_ylim([1e-3, 1e1])  # Limit y-axis from 0.1 to 1
ax.legend()
# Set integer tick marks on x-axis
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

# Save and show plot
ps.set(ax)
plt.savefig("2D_vortex_Err_CPU.pdf", transparent=False)
plt.show()

