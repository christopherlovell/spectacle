import pickle as pcl
import os
import numpy as np
from weights import calculate_weights
import astropy.units as u
from astropy.cosmology import z_at_value

class spectra:

    def create_lookup_table(self, z, resolution=5000):

        # if query_yes_no("Lookup table not initialised. Would you like to do this now? (takes a minute or two)"):

        lookback_time = self.cosmo.lookback_time(z).value # Gyr

        self.age_lookup = np.linspace(lookback_time, lookback_time + self.age_lim, resolution)
        self.a_lookup = np.array([self.cosmo.scale_factor(z_at_value(self.cosmo.lookback_time, a * u.Gyr)) for a in self.age_lookup], dtype=np.float32)

        filename = "%s/lookup_tables/lookup_%s_z%03.0fp%.3s_lim%1.0fp%.3s.txt"%(self.grid_directory,
                                                                       self.cosmo.name, 
                                                                       z, str(z%1)[2:], 
                                                                       self.age_lim,
                                                                       str("%.3f"%(self.age_lim%1))[2:])

        np.savetxt(filename, np.array([self.a_lookup, self.age_lookup]))


    def load_lookup_table(self, z):

        filename = "%s/lookup_tables/lookup_%s_z%03.0fp%.3s_lim%1.0fp%.3s.txt"%(self.grid_directory,
                                                                       self.cosmo.name, 
                                                                       z, str(z%1)[2:],
                                                                       self.age_lim,
                                                                       str("%.3f"%(self.age_lim%1))[2:])

        if os.path.isfile(filename):
            lookup_table = np.loadtxt(filename, dtype=np.float32)
            self.a_lookup = lookup_table[0]
            self.age_lookup = lookup_table[1]
        else:
            print("lookup table not initialised for this cosmology / redshift / age cutoff. initialising now (make take a couple of minutes)")
            self.create_lookup_table(z)



    def load_grid(self,name='bc03_chab', z0=0.0, grid_directory='',verbose=False):
        """
        Load SPS model
        """
        grid_directory = '/research/astro/highz/Students/Chris/sph2sed/grids'
        file_dir = '%s/intrinsic/output/%s.p'%(grid_directory,name)

        if verbose: print("Loading %s model from: \n\n%s\n"%(name, file_dir))
        temp = pcl.load(open(file_dir, 'rb'))

        grid = {'name': name, 'grid': None, 'age': None, 'metallicity':None}
        grid['grid'] = temp['Spectra']
        grid['metallicity'] = temp['Metallicity']
        grid['age'] = {z0: temp['Age']}  # scale factor
        grid['lookback_time'] = {z0: self.cosmo.lookback_time((1. / temp['Age']) - 1).value}  # Gyr
        grid['age_mask'] = {z0: np.ones(len(temp['Age']), dtype='bool')}
        grid['wavelength'] = temp['Wavelength']

        ## Sort grids
        if grid['age'][z0][0] > grid['age'][z0][1]:
             if verbose: print("Age array not sorted ascendingly. Sorting...\n")
             grid['age'][z0] = grid['age'][z0][::-1]
             grid['age_mask'][z0] = grid['age_mask'][z0][::-1]
             grid['lookback_time'][z0] = grid['lookback_time'][z0][::-1]
             grid['grid'] = grid['grid'][:,::-1,:]


        if grid['metallicity'][0] > grid['metallicity'][1]:
            if verbose: print("Metallicity array not sorted ascendingly. Sorting...\n")
            grid['metallicity'] = grid['metallicity'][::-1]
            grid['grid'] = grid['grid'][::-1,:,:]


        return grid



    def redshift_grid(self,grid, z, z0=0.0):
        """
        Redshift grid ages, return new grid
        Args:
            z (float) redshift
        """

        if z == z0:
            print("No need to initialise new grid, z = z0")
            return grid
        else:
            observed_lookback_time = self.cosmo.lookback_time(z).value
            # if verbose: print("Observed lookback time: %.2f"%observed_lookback_time)

            # redshift of age grid values
            age_grid_z = [z_at_value(self.cosmo.scale_factor, a) for a in grid['age'][z0]]
            # convert to lookback time
            age_grid_lookback = np.array([self.cosmo.lookback_time(z).value for z in age_grid_z])
            # add observed lookback time
            age_grid_lookback += observed_lookback_time

            # truncate age grid by age of universe
            age_mask = age_grid_lookback < self.cosmo.age(0).value
            age_mask = age_mask & ~(np.isclose(age_grid_lookback, self.cosmo.age(0).value))
            age_grid_lookback = age_grid_lookback[age_mask]

            # convert new lookback times to redshift
            age_grid_z = [z_at_value(self.cosmo.lookback_time, t * u.Gyr) for t in age_grid_lookback]
            # convert redshift to scale factor
            age_grid = self.cosmo.scale_factor(age_grid_z)

            grid['age'][z] = age_grid
            grid['lookback_time'][z] = age_grid_lookback - observed_lookback_time
            grid['age_mask'][z] = age_mask

            return grid



    def weights_grid(self,shid,Z,A,resample=False):
        """
        Calculate SPS weights
        """

        ftime,met,imass = self.load_galaxy(shid)

        if resample:
            dat = self.resample_recent_sf(ftime,met,imass)
            
            if dat is None:
                in_arr = np.array([met,ftime,imass],
                              dtype=np.float64)
            else:
                rftime = dat[0]
                rmet = dat[1]
                rimass = dat[2]
                mask = dat[3]
    
                in_arr = np.array([np.hstack([met[~mask],rmet]),
                                   np.hstack([ftime[~mask],rftime]),
                                   np.hstack([imass[~mask],rimass])],
                                  dtype=np.float64)
        else:
            in_arr = np.array([met,ftime,imass],
                              dtype=np.float64)
    
        weights_temp = calculate_weights(Z, A, in_arr)
        return weights_temp



    def calc_intrinsic_spectra(self,weights,grid,z=0.0):
        """
        return intrinsic spectra
        """
        grid = grid['grid'][:,grid['age_mask'][z],:]
        return np.einsum('ijk,jkl->il',weights,grid) #,optimize=True)



    def two_component_dust(self, weights, grid, z=0.0, lambda_nu=5500, tau_ism=0.33, tau_cloud=0.67, tdisp=1e-2, gamma=0.7, gamma_cloud=0.7):


        grid_temp = grid['grid'][:,grid['age_mask'][z],:]
        normed_wl = grid['wavelength'] / lambda_nu
        lookback = grid['lookback_time'][z]

        spec_A = np.einsum('ijk,jkl->il',
                           weights[:,:,lookback < tdisp],
                           grid_temp[:,lookback < tdisp,:])
                           #,optimize=True)

        T = np.exp(-1 * (tau_ism + tau_cloud) * normed_wl**-gamma)
        spec_A *= T

        spec_B = np.einsum('ijk,jkl->il',
                           weights[:,:,lookback >= tdisp],
                           grid_temp[:,lookback >= tdisp,:],
                           optimize=True)

        T = np.exp(-1 * tau_ism * normed_wl**-gamma_cloud)
        spec_B *= T

        dust_spectra = spec_A + spec_B
        return dust_spectra



    @staticmethod
    def simple_metallicity_factor(sf_gas_metallicity, sf_gas_mass, tau_0=1e-8):
        Z_Factor = sf_gas_metallicity * sf_gas_mass * tau_0
        return Z_factor[:,None]  # add an axis

    
    @staticmethod
    def metallicity_factor_update(sf_gas_metallicity, sf_gas_mass, stellar_mass, 
                                  gas_norm=2, metal_norm=0.0015, 
                                  alpha=1, beta=1, tau_0=1):

        Z_factor = tau_0 * (sf_gas_metallicity / metal_norm)**alpha * ((sf_gas_mass / stellar_mass) / gas_norm)**beta
        return Z_factor[:,None]  # add an axis
    
    
    def tau_0(self):
        z = self.redshift
        MW_gas_fraction = 0.1
        milkyway_mass = 10.8082109
        Z_solar = 0.0134
        M_0 = np.log10(1 + z) * 2.64 + 9.138
        Z_0 = 9.102
        beta = 0.513
        
        # Zahid+14 Mstar - Z relation (see Trayford+15)
        logOHp12 = Z_0 + np.log(1 - np.exp(-1 * (10**(milkyway_mass - M_0))**beta))
        
        # Convert from 12 + log()/H) -> Log10(Z / Z_solar)
        # Allende Prieto+01 (see Schaye+14, fig.13)
        Z = 10**(logOHp12 - 8.69)

        return 1 / (Z * 0.0134) * (1/MW_gas_fraction)



    @staticmethod
    def interp_negative(spec, wl):
        for i,_s in enumerate(spec):
            mask = _s < 0
            vals = np.interp(wl[mask],wl[~mask],_s[~mask])
            spec[i][mask] = vals

        return spec


    def add_noise_flat(self, spec, wl, sn=50):
        """
        Add flat noise to spectrum
        """
        noise = np.random.normal(loc=0, scale=spec / sn)
        noisified_spectra = spec + noise

        # interpolate negative values
        return self.interp_negative(noisified_spectra, wl)


