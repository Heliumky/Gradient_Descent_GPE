import matplotlib.pyplot as plt
import plotsetting as ps

fig,ax = plt.subplots()
ax.plot([1,2],[1,2],marker='o')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ps.set(ax)
plt.show()
