import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots()

# Store Fano indices
fidx = []

# Read file from cleaned dataset of probable examples
with open('../data/terminal_dim8_probable.txt', 'r') as f:
    
   # Extract only the Fano indices, the line looks like 'FanoIndes: 1'
   for x in f:
       x = x.split(':')
       if 'FanoIndex' in x[0]:
           fidx.append(int(x[1])) 

# Histogram
bins = range(min(fidx),max(fidx))

# Seaborn histogram
g = sns.histplot(fidx, bins=bins, edgecolor="white", color="grey")

# Log the y-axis
g.set_yscale('log')

# Set axes labels
ax.set_xlabel(r'Fano index', fontsize = 17)
ax.set_ylabel(r'Log count', fontsize = 17)

# Position the y-axis at x=0
plt.xlim(xmin=0)

# Save the images
for dpi in [300,600,1200]:
    plt.savefig(f'../images_{dpi}/indices_distribution.png', dpi=dpi, bbox_inches='tight')