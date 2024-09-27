import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import plotsetting as ps

# Load data (example data)
intelf = np.loadtxt("2d_intel_time.txt")
tdvp = np.loadtxt("2d_TDVP_time.txt")

# Create figure and axis with adjusted size
fig, ax = plt.subplots()

# Scatter plot for Intel Fortran and one-site TDVP
ax.scatter(intelf[:,0], intelf[:,1], marker='x', label="traditional", color='black', s=100)
ax.scatter(tdvp[:,0], tdvp[:,1], marker='.', label="QTT", color='red', s=100)

# Linear fit to one-site TDVP data
coefficients = np.polyfit(tdvp[:,0][3:], tdvp[:,1][3:], deg=1)
poly = np.poly1d(coefficients)
tdvp_fit = np.linspace(min(tdvp[:,0]), max(tdvp[:,0]), 100)
ax.plot(tdvp_fit, poly(tdvp_fit), color='red', linewidth=2, linestyle='--')

# Linear fit to Foftran data
coefficients_ft = np.polyfit(intelf[:,0], np.log(intelf[:,1]), deg=1)
poly_ft = np.poly1d(coefficients_ft)
ft_fit = np.linspace(min(intelf[:,0]), 12, 100)  # Adjusted range for better visibility
ax.plot(ft_fit, np.exp(poly_ft(ft_fit)), color='black', linewidth=2, linestyle='--')

# Set labels and title
ax.set_xlabel(r'number of qubits')
ax.set_ylabel(r'CPU time [s]')
ax.set_yscale('log')
ax.set_ylim([1e-5, 3*1e-1])  # Limit y-axis from 0.1 to 1

# Set integer tick marks on x-axis
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

# Function to convert x values for secondary axis
def pow2(x):
    return 2**x

# Create inset axes for linear y-axis data
axins = inset_axes(ax, width="40%", height="40%",  bbox_to_anchor=(-0.25, -0.3, 1.2, 1.2), bbox_transform=ax.transAxes)
axins.scatter(intelf[:,0], intelf[:,1], marker='x', label="traditional", color='black', s=50)
axins.scatter(tdvp[:,0], tdvp[:,1], marker='.', label="QTT", color='red', s=50)
axins.plot(tdvp_fit, poly(tdvp_fit), color='red', linewidth=1, linestyle='--')
axins.plot(ft_fit, np.exp(poly_ft(ft_fit)), color='black', linewidth=1, linestyle='--')

# Set labels and scale for inset axes
axins.set_xlabel(r'number of qubits',fontsize=8)
axins.set_ylabel(r'CPU time [s]',fontsize=8)
#axins.set_xscale('log')
axins.set_yscale('linear')
axins.set_xlim(ax.get_xlim())  # Match x-axis limits with main plot
axins.set_ylim(ax.get_ylim())

axins.tick_params(axis='both', which='major', labelsize=8)
axins.tick_params(axis='both', which='minor', labelsize=8)
# Hide y-axis labels/ticks for inset plot
axins.yaxis.set_visible(True)
axins.xaxis.set_major_locator(MaxNLocator(integer=True))
# Add legend to main plot
ax.legend()

x = tdvp[:,0]
#print(x)
def pow2(x):
    x = np.array(x, np.int64)  # Convert input to a numpy array of floats
    x = 2**x
    return x
discrete_value = pow2

# Create secondary x-axis at the top
secax = ax.secondary_xaxis('top', functions=(pow2, discrete_value))
secax.set_xlabel('discretization points')
secax.set_xscale('log')

# Save and show plot
ps.set(ax)
plt.savefig("2D_log_inset.pdf", transparent=False)
plt.show()

