from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz


def alt_fromloc_atdate(obj_name, lat, lon, alt, tw_ev, tw_mo):

    location = EarthLocation(lat = lat * u.deg, lon = lon * u.deg, height = alt * u.m)

    target = Simbad.query_object(obj_name)
    RA, DEC = target['RA'][0], target['DEC'][0]
    coord_tar = SkyCoord(f'{RA} {DEC}', unit = (u.hourangle, u.deg))

    t = np.arange(tw_ev - 1/24, tw_mo + 1/24, step = 5/(24 * 60)) # time array with steps of 5 min with +- 1 h of twilights
    alt_tar = coord_tar.transform_to(AltAz(location = location, obstime = Time(t, format = 'jd'))).alt.value

    return alt_tar