import numpy as np
from astropy.coordinates import EarthLocation, AltAz, get_sun
from astropy.time import Time
import astropy.units as u


def range_tw(jd_arr, location, which_tw = 'both'):

    alt_astronomical_tw = -18 # deg

    range_tw_mo = []
    range_tw_ev = []
    for step,jdi in enumerate(jd_arr):
        time_format = Time(jdi, format = 'jd', scale = 'utc')
        sun_coord = get_sun(time_format).transform_to(AltAz(obstime = time_format, location = location)) # Alt, Az coordinates of the sun
        sun_alt = sun_coord.alt.value
        dif = sun_alt - alt_astronomical_tw
        if step == 0:
            prev_dif = dif
            continue # in the first step we just need to define the variable 'prev_dif' and the we continue with the 'for' loop
        if dif * prev_dif < 0: # check if there is a change in the sign of 'dif' in this step
            if dif > 0: # the altitude of the sun is increasing and therefore this is the morning twilight
                range_tw_mo = [jd_arr[step - 1], jdi]
            else: # the altitude of the sun is decreasing and therefore this is the evening twilight
                range_tw_ev = [jd_arr[step - 1], jdi]

        # check if the target twilight has been found. If so, there is no need to continue the loop
        if (len(range_tw_mo) != 0) and (len(range_tw_ev) != 0) and (which_tw == 'both'):
            break
        elif (len(range_tw_mo) != 0) and (which_tw == 'mo'):
            break
        elif (len(range_tw_ev) != 0) and (which_tw == 'ev'):
            break
        
        prev_dif = dif # redifine for the next step

    return range_tw_mo, range_tw_ev


def compute_tw(date, lat, lon, alt): # the date must be in format 'Y/M/D'
    
    location = EarthLocation(lat = lat * u.deg, lon = lon * u.deg, height = alt * u.m)

    # first approximation of the twilightls by using big steps of 1 h
    jd_init = Time(date + 'T00:00:00', scale = 'utc', format = 'isot').jd # beginning of the day
    jd_end = jd_init + 1.0 # end of the day
    jd_h_step = np.linspace(jd_init, jd_end, 25)
    range_tw_mo, range_tw_ev = range_tw(jd_h_step, location)

    # we fine tune the twilightls by using smaller steps of 1 min
    jd_tw_mo_array = np.linspace(range_tw_mo[0], range_tw_mo[1], 61) # array with steps of 1 min
    small_range_tw_mo, _ = range_tw(jd_tw_mo_array, location, which_tw = 'mo')
    tw_mo = np.median(small_range_tw_mo) # as an approximation. The uncertainty is 1 min
    
    jd_tw_ev_array = np.linspace(range_tw_ev[0], range_tw_ev[1], 61) # array with steps of 1 min
    _, small_range_tw_ev = range_tw(jd_tw_ev_array, location, which_tw = 'ev')
    tw_ev = np.median(small_range_tw_ev)

    return tw_mo, tw_ev


if __name__ == '__main__':

    latitude, longitude, altitude = [40.441401, -3.952766, 665] # Coordinates in units [deg], [deg], [m
    
    date = Time.now().isot[:10]
    _, time_tw_ev = compute_tw(date, latitude, longitude, altitude)

    print(f'The evening twilight today at ESAC is at {Time(time_tw_ev, format = "jd", scale = "utc").isot[11:16]} UTC')