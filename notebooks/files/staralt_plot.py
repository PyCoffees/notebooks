import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import AutoMinorLocator
from astropy.time import Time
import matplotlib.dates as mdates


rcParams['font.size'] = 16.0
rcParams['axes.linewidth'] = 1

rcParams['ytick.major.size'] = 6
rcParams['ytick.major.width'] = 1.5
rcParams['ytick.minor.size'] = 3
rcParams['ytick.minor.width'] = 0.5

rcParams['xtick.major.size'] = 6
rcParams['xtick.major.width'] = 1.5
rcParams['xtick.minor.size'] = 3
rcParams['xtick.minor.width'] = 0.5

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica']


def do_plot(tw_ev, tw_mo, alt_tar, obj_name, obs_name):

    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (9, 5))

    # main plot
    x_utc = pd.date_range(start = Time(tw_ev - 1/24, format = 'jd').isot, end = Time(tw_mo + 1/24, format = 'jd').isot, periods = len(alt_tar))
    ax.plot(x_utc, alt_tar, lw = 3, color = 'k', label = f'{obj_name} from {obs_name} ')
    
    # mark the twilights
    ax.fill_between([pd.to_datetime(Time(tw_mo, format = 'jd').isot), x_utc[-1]], -10, 100, color = 'blue', alpha = 0.4)
    ax.fill_between([x_utc[0], pd.to_datetime(Time(tw_ev, format = 'jd').isot)], -10, 100, color = 'blue', alpha = 0.4)

    # labels, limits of the axis, title and others
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel('UTC Time')
    ax.set_ylabel(f'Altitude [deg]')
    ax.grid(True, alpha = 0.2)

    ax.set_ylim(0, 90)
    ax.set_xlim(pd.to_datetime(Time(tw_ev - 1/24, format = 'jd').isot), pd.to_datetime(Time(tw_mo + 1/24, format = 'jd').isot))

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params('both', direction = 'in', length = 10, width = 1.5, which = 'major', labelsize = 15)
    ax.tick_params('both', direction = 'in', length = 5, width = 0.5, which = 'minor')
    
    date_init = Time(tw_ev, format = 'jd', scale = 'utc').isot[:10]
    day_end = Time(tw_mo, format = 'jd', scale = 'utc').isot[8:10]
    date = date_init +' / '+ day_end
    ax.set_title(f'{date}', fontsize = 18)

    plt.legend(loc = 'upper right', fontsize = 12)
    plt.show()
