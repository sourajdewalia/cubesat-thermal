from skyfield.api import Topos, load, EarthSatellite, Time, Timescale
from sgp4.api import Satrec, WGS72, jday
from math import sin, cos, pi
import numpy as np
import matplotlib.pyplot as plt



class SatelliteFlux():
    def __init__(self, whichconst=WGS72, opsmode='i', satnum=5, epoch=None, bstar=2.8098e-05,
                 ndot=6.969196665e-13, nddot=0.0, ecco=0.1859667, argpo=5.7904160274885,
                 inclo=0.5980929187319, mo=0.3373093125574, no_kozai=0.0472294454407, nodeo=6.0863854713832,
                 orientation='sun'):
        self.ts = load.timescale()
        self.t_now = self.ts.now()
        self.t_origin = 2433281.5
        self.t_days_now = self.t_now.ut1

        if epoch == None:
            epoch = (self.t_days_now - self.t_origin) + 0.5
        else:
            epoch = (epoch - self.t_origin) + 0.5

        self.whichconst  = whichconst # gravity model
        self.opsmode     = opsmode    # 'a' = old AFSPC mode, 'i' = improved mode
        self.satnum      = satnum     # satnum: Satellite number
        self.epoch       = epoch      # epoch: days since 1949 December 31 00:00 UT
        self.bstar       = bstar      # bstar: drag coefficient (/earth radii)
        self.ndot        = ndot       # ndot: ballistic coefficient (revs/day)
        self.nddot       = nddot      # nddot: second derivative of mean motion (revs/day^3)
        self.ecco        = ecco       # ecco: eccentricity
        self.argpo       = argpo      # argpo: argument of perigee (radians)
        self.inclo       = inclo      # inclo: inclination (radians)
        self.mo          = mo         # mo: mean anomaly (radians)
        self.no_kozai    = no_kozai   # no_kozai: mean motion (radians/minute)
        self.nodeo       = nodeo      # nodeo: right ascension of ascending node (radians)

        self.satrec = Satrec()
        self.satrec.sgp4init(
            self.whichconst, self.opsmode, self.satnum, self.epoch, self.bstar, self.ndot,
            self.nddot, self.ecco, self.argpo, self.inclo, self.mo, self.no_kozai, self.nodeo
            )
        self.satellite = EarthSatellite.from_satrec(self.satrec, self.ts)
        self.planets   = load('de421.bsp')
        self.earth = self.planets['earth']
        self.sun   = self.planets['sun']
        self.ssb_satellite = self.earth + self.satellite
        self.orientation = orientation

    # Get flux vector values for time after epoch (in seconds)
    def get_orientation_flux_vector(self, t):
        orient_vec, flux_vec = np.zeros((3,3)), np.zeros(3)
        t0 = self.satellite.epoch
        t_req = self.ts.utc(year=t0.utc.year, month=t0.utc.month, day=t0.utc.day, hour=t0.utc.hour,
                    minute=t0.utc.minute, second=t0.utc.second + t)

        #get satellite position in earth frame
        sat_position = self.satellite.at(t_req)
        sat_position_abs = self.ssb_satellite.at(t_req)
        earth_position = self.earth.at(t_req)
        
        if self.orientation == 'sun':
            sun_position = self.sun.at(t_req)
            sat_dir_vec = (sun_position - sat_position_abs).position
            earth_dir_vec = (earth_position - sat_position_abs).position
            sat_velocity_vec = sat_position_abs.velocity.km_per_s
            # print('sat dir', sat_dir_vec.km)
            # print('sat vel', sat_velocity_vec)
            # print('earth dir', earth_dir_vec)
            
            x_orient = sat_dir_vec.km/sat_dir_vec.length().km
            
            temp_vec_y = np.cross(np.cross(sat_velocity_vec, earth_dir_vec.km), x_orient)
            y_orient = temp_vec_y/np.sqrt(np.dot(temp_vec_y,temp_vec_y))
            
            temp_vec_z = np.cross(x_orient, y_orient)
            z_orient = temp_vec_z/np.sqrt(np.dot(temp_vec_z,temp_vec_z))

            orient_vec[0,:] = x_orient
            orient_vec[1,:] = y_orient
            orient_vec[2,:] = z_orient

            if sat_position.is_sunlit(self.planets):
                flux_vec[:] = -x_orient

        elif self.orientation == 'earth':
            sat_dir_vec = (earth_position - sat_position_abs).position
            earth_dir_vec = sat_dir_vec
            sat_velocity_vec = sat_position_abs.velocity.km_per_s
            
            x_orient = sat_dir_vec.km/sat_dir_vec.length().km
            
            temp_vec_y = np.cross(np.cross(sat_velocity_vec, earth_dir_vec.km), x_orient)
            y_orient = temp_vec_y/np.sqrt(np.dot(temp_vec_y,temp_vec_y))
            
            temp_vec_z = np.cross(x_orient, y_orient)
            z_orient = temp_vec_z/np.sqrt(np.dot(temp_vec_z,temp_vec_z))

            orient_vec[0,:] = x_orient
            orient_vec[1,:] = y_orient
            orient_vec[2,:] = z_orient
            if sat_position.is_sunlit(self.planets):
                sun_position = self.sun.at(t_req)
                dir_vec = (sat_position_abs - sun_position).position
                flux_vec[:] = dir_vec.km/dir_vec.length().km

        return orient_vec, flux_vec

    def _plot(self, tf, dt=60.):
        t_sec_idx = np.arange(0., tf, dt)
        t_sec_idx = np.append(t_sec_idx, tf)
        t = self.satellite.epoch
        fig, ax = plt.subplots()
        t_idx = self.ts.utc(year=t.utc.year, month=t.utc.month, day=t.utc.day, hour=t.utc.hour,
                minute=t.utc.minute, second=t.utc.second + t_sec_idx)
        sat_position = self.satellite.at(t_idx)
        dst = sat_position.distance().km
        shadow = (sat_position.is_sunlit(self.planets)).astype(int)
        ax.plot(t_sec_idx, dst, color='black')
        ax.fill_between(t_sec_idx, max(dst), min(dst), where=[not s for s in shadow],
                        color='green', alpha=0.5)#, transform=ax.get_xaxis_transform())
        #ax.plot(t_sec_idx, np.ones_like(dst)*6371,'--', color='red')
        plt.xlabel('t (sec.)')
        plt.ylabel('Distance (km)')
        plt.legend()
        plt.grid()
        #plt.show()
        plt.savefig("orbit.png", dpi=300)
        plt.close()
        #print(shadow)
# From a place on Earth (Topocentric)

# boston = earth + Topos('42.3583 N', '71.0603 W')
# astrometric = boston.at(t).observe(mars)
# apparent = boston.at(t).observe(mars).apparent()
if __name__ == '__main__':
    #app = SatelliteFlux(epoch=2433282, no_kozai=0.07, ecco=0.02)
    app = SatelliteFlux(epoch=2433282)
    app._plot(20000, dt=60)