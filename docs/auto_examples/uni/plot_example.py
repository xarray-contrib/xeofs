"""
EOF analysis (S-mode)
===========================

EOF analysis in S-mode (maximising temporal variance)
"""

import xarray as xr
import matplotlib.pyplot as plt

da = xr.tutorial.load_dataset('air_temperature')['air']
da.isel(time=0).plot()
plt.savefig('blaa.jpg')
