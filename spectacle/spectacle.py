#!/usr/bin/python

import pickle as pcl
import sys
import os
import numpy as np
import h5py
# from weights import calculate_weights
import schwimmbad
import astropy.units as u
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
from functools import partial

if __package__ is None or __package__ == '':
    # uses current directory visibility
    import spectres
    import _spectra,_photo
else:
    # uses current package visibility
    from spectacle import spectres
    from spectacle import _spectra,_photo

class spectacle(_spectra.spectra, _photo.photo):
    """
    Class encapsulating data structures and methods for generating spectral 
    energy distributions (SEDs) from cosmological hydrodynamic simulations.
    """

    def __init__(self, fname='../derivedSFH/data/full_histories_illustris_hmass.h5', details=''):
        self.package_directory = os.path.dirname(os.path.abspath(__file__))          # location of package
        self.grid_directory = os.path.split(self.package_directory)[0]+'/grids'      # location of SPS grids
        self.filter_directory = os.path.split(self.package_directory)[0]+'/filters'  # location of filters
        self.filename = fname
        self.cosmo = cosmo     # astropy cosmology
        self.age_lim = 0.1     # Young star age limit, used for resampling recent SF, Gyr
        with h5py.File(self.filename,'r') as f:
            self.redshift = f.attrs['redshift']

        
        # create temp grid directory
        path = "%s/temp"%self.package_directory
        if not os.path.isdir(path):
            try:
                os.mkdir(path)
            except OSError:  
                print ("Creation of the /temp directory failed")


    def refresh_directories(self):
        self.package_directory = os.path.dirname(os.path.abspath(__file__))      # location of package
        self.grid_directory = os.path.split(self.package_directory)[0]+'/grids'  # location of SPS grids


    def load_galaxy(self,shidx):
        """
        Load particle data for a single galaxy from an hdf5 file
        """
        with h5py.File(self.filename, 'r') as f:
            midx = np.where(shidx == f['Subhalos/ID'][:])[0][0]
        
            pidx = f['Subhalos/Index Start'][midx]
            plen = f['Subhalos/Index Length'][midx]
    
            ftime = f['Star Particles/Formation Time'][pidx:(pidx+plen)]
            met = f['Star Particles/Metallicity'][pidx:(pidx+plen)]
            imass = f['Star Particles/Initial Mass'][pidx:(pidx+plen)]
    
        
        return (ftime,met,imass)
    

    def resample_recent_sf(self, ftime, met, imass, verbose=False):
        """
        Resample recently formed star particles.
        Star particles are much more massive than individual HII regions, leading to artificial Poisson scatter in the SED from recently formed particles.
        Args:
            idx (int) galaxy index
            age_lim (float) cutoff age in Gyr, lookback time
        """

        if ('a_lookup' not in self.__dict__) | ('age_lookup' not in self.__dict__):
            self.load_lookup_table(self.redshift)

        if (self.a_lookup.min() > self.cosmo.scale_factor(self.redshift)) |\
                (self.a_lookup.max() < self.cosmo.scale_factor(self.redshift)):
            if verbose: print('Lookup table out of range. Reloading')
            self.load_lookup_table(self.redshift)
       

        lookback_time_z0 = np.float32(self.cosmo.lookback_time(self.redshift).value)
        lookback_time_z1 = np.float32((self.cosmo.lookback_time(self.redshift)\
                + self.age_lim * u.Gyr).value)

        # find age_cutoff in terms of the scale factor
        self.age_cutoff = self.cosmo.scale_factor(z_at_value(self.cosmo.lookback_time, \
                lookback_time_z1 * u.Gyr))

        mask = ftime > self.age_cutoff
        N = np.sum(mask)

        if N > 0:
            lookback_times = self.cosmo.lookback_time((1. / ftime[mask]) - 1).value
            if verbose: print("Young stellar particles: %s"%N)
        else:
            if verbose: print("No young stellar particles!")
            return None

        resample_ages = np.array([], dtype=np.float32)
        resample_mass = np.array([], dtype=np.float32)
        resample_metal = np.array([], dtype=np.float32)
        
        for p_idx in np.arange(N):
       
            n = int(imass[mask][p_idx] / 1e4)
            M_resample = np.float32(imass[mask][p_idx] / n)
        
            new_lookback_times = np.random.uniform(lookback_time_z0,lookback_time_z1,size=n)
       
            # lookup scale factor in tables 
            resample_ages = np.append(resample_ages, 
                                      self.a_lookup[np.searchsorted(self.age_lookup, new_lookback_times)])
        
            resample_mass = np.append(resample_mass, np.repeat(M_resample, n))
        
            resample_metal = np.append(resample_metal, 
                                np.repeat(met[mask][p_idx], n))
           

        return [resample_ages,resample_metal,resample_mass,mask]



    def recalculate_sfr(self, shid, time=0.1):
        """
        Recalculate SFR using particle data. 
        Adds an entry to the galaxies dict with key `label`.
        Args:
            shid (int) galaxy index
            time (float) lookback time over which to calculate SFR, Gyr
        """

        ftime,met,imass = self.load_galaxy(shid)

        # find age limit in terms of scale factor
        scalefactor_lim = self.cosmo.scale_factor(z_at_value(self.cosmo.lookback_time, (self.cosmo.lookback_time(self.redshift).value + time) * u.Gyr))

        # mask particles below age limit
        mask = ftime > scalefactor_lim

        # sum mass of particles (Msol), divide by time (yr)
        return np.sum(imass[mask]) / (time * 1e9)  # Msol / yr

 
    
    def save_spectra(self, name, spectra, wavelength, **kwargs):
        """
        Save spectra to file
        """
        with h5py.File(self.filename, 'a') as f:
        
            if "Spectra" not in list(f.keys()): f.create_group("Spectra") 
            
            s1 = len(spectra)
            s2 = spectra.shape[1]
    
            if name not in list(f["Spectra"].keys()):
                f["Spectra"].create_group(name)

            if 'spectrum' not in list(f["Spectra/%s"%name].keys()):
                f["Spectra/%s"%name].create_dataset("spectrum",spectra.shape,
                                                    maxshape=(None,s2),data=spectra)
            else:
                f["Spectra/%s/spectrum"%name].resize(spectra.shape)
                f["Spectra/%s/spectrum"%name][:] = spectra


            if 'wavelength' not in list(f["Spectra/%s"%name].keys()):
                f["Spectra/%s"%name].create_dataset("wavelength",wavelength.shape,
                                                    maxshape=(None,),data=wavelength)
            else:
                f["Spectra/%s/wavelength"%name].resize(wavelength.shape)
                f["Spectra/%s/wavelength"%name][:] = wavelength

            for kw,arg in kwargs.items():
                f["Spectra/%s"%name].attrs[str(kw)] = arg
    
    
    def _clear_spectra(self):
        """
        Save spectra to file
        """
        with h5py.File(self.filename, 'a') as f:
            if "Spectra" in list(f.keys()):
                del f["Spectra"] 


    def load_spectra(self, name):
        """
        Load spectra from file
        """
        with h5py.File(self.filename, 'r') as f:
        
            if "Spectra" not in list(f.keys()):
                print("No spectra...")
                return None
    
            spectra = f["Spectra/%s/spectrum"%name][:]
            wavelength = f["Spectra/%s/wavelength"%name][:]

        return spectra, wavelength


    def save_arr(self, arr, name, group='', replace=True):
        """
        Load Subhalo array from file
        """
        maxshape = tuple([None] * arr.ndim)
        
        with h5py.File(self.filename, 'a') as f:

            if group not in f:
                f.create_group(group)

            if name not in list(f['/%s'%group].keys()):
                f["/%s"%group].create_dataset(name, shape=arr.shape,
                                             maxshape=maxshape,data=arr)
            else:
                if replace:
                    if len(arr) == len(f["/%s/%s"%(group,name)][:]):
                        f["/%s/%s"%(group,name)][:] = arr
                    else:
                        raise ValueError('Error: New data shape not equal to existing shape')
                # raise ValueError('Key already in Subhalos group!')



    def load_arr(self, name, group=''):
        """
        Load Subhalo array from file
        """
        with h5py.File(self.filename, 'r') as f:

            if group not in f:
                print("Group doesn't exist...")
                return None

            if name not in list(f['/%s'%group].keys()):
                print("'%s' Dataset doesn't exist..."%name)
                return None

            arr = f["/%s/%s"%(group,name)][:]

        return arr

    
    def rebin_spectra(self, spec, wl, resample_wl):
        """
        Resample spectrum on a new wavelength grid using the `spectres` package

        Args:
            spec: (array[N,l]) spectra
            wl: (array[l])
            resample_wl (array[L]) l

        Returns:
            resamp_spec (array[N,L])
        """
        resamp_spec = spectres.spectres(resampling=resample_wl, 
                               spec_fluxes=spec.T, 
                               spec_wavs=wl).T

        return resamp_spec



# def main(pool):
#     tacle = spectacle(fname='../../derivedSFH/data/full_histories_eagle.h5')
#     tacle._clear_spectra()
#     # N = 10
#     # print('N:',N)
# 
#     with h5py.File(tacle.filename, 'r') as f:
#         shids = f['Subhalos/ID'][:]# [:N]
# 
# 
#     # ## SFR timescales (in parallel)
#     # print('SFR 100:')
#     # lg = partial(tacle.recalculate_sfr, time=0.1)
#     # sfr100 = np.array(list(pool.map(lg,shids)))
#     # 
#     # print('SFR 10:')
#     # lg = partial(tacle.recalculate_sfr, time=0.01)
#     # sfr10 = np.array(list(pool.map(lg,shids)))
#     # 
#     # print('SFR 50:')
#     # lg = partial(tacle.recalculate_sfr, time=0.05)
#     # sfr50 = np.array(list(pool.map(lg,shids)))
#     # 
#     # print('SFR 500:')
#     # lg = partial(tacle.recalculate_sfr, time=0.5)
#     # sfr500 = np.array(list(pool.map(lg,shids)))
#     # 
#     # print('SFR 1000:')
#     # lg = partial(tacle.recalculate_sfr, time=1.0)
#     # sfr1000 = np.array(list(pool.map(lg,shids)))
#     # pool.close()
# 
#     # tacle.save_arr(sfr100,'SFR 100Myr','Subhalos')
#     # tacle.save_arr(sfr10,'SFR 10Myr','Subhalos')
#     # tacle.save_arr(sfr50,'SFR 50Myr','Subhalos')
#     # tacle.save_arr(sfr500,'SFR 500Myr','Subhalos')
#     # tacle.save_arr(sfr1000,'SFR 1Gyr','Subhalos')
# 
#     print('z:',tacle.redshift)
#     grid = tacle.load_grid(name='fsps_neb', grid_directory=tacle.grid_directory)
#     grid = tacle.redshift_grid(grid,tacle.redshift)
#     Z = grid['metallicity']
#     A = grid['age'][tacle.redshift]
#     wl = grid['wavelength']
# 
#     ## Calculate weights (in parallel)
#     lg = partial(tacle.weights_grid, Z=Z, A=A, resample=True)
#     weights = np.array(list(pool.map(lg,shids)))
#     pool.close()
# 
# 
#     ## Dust metallicity factor
#     with h5py.File(tacle.filename, 'r') as f:
#         shids = f['Subhalos/ID'][:]# [:N]
#         sfgmass = f["Subhalos/Star Forming Gas Mass"][:]# [:N]
#         sm30 = 10**f["Subhalos/Stellar Mass 30kpc"][:]# [:N] 
#         sfgmet = f["Subhalos/Stellar Metallicity 30kpc"][:]# [:N]
# 
#     tau_0 = tacle.tau_0()
#     print("tau_0:",tau_0)
#     Z_factor = tacle.metallicity_factor_update(sfgmet,sfgmass,sm30,gas_norm=1,metal_norm=1,tau_0=290)#tau_0)
# 
#     ## Calculate intrinsic and two-component dust attenuated spectra
#     intrinsic_spectra = tacle.calc_intrinsic_spectra(weights,grid,z=tacle.redshift)
#     dust_spectra = tacle.two_component_dust(weights,grid,z=tacle.redshift,
#                                             tau_ism=0.33 * Z_factor, tau_cloud=0.67 * Z_factor)
#    
#     ## Rebin spectra
#     resample_wl = np.loadtxt('../../derivedSFH/data/vespa_lambda.txt')
#     intrinsic_spectra = tacle.rebin_spectra(intrinsic_spectra, grid['wavelength'],resample_wl)
#     dust_spectra = tacle.rebin_spectra(dust_spectra, grid['wavelength'],resample_wl)
#  
#     ## Save to HDF5
#     tacle.save_spectra('Intrinsic', intrinsic_spectra, resample_wl, units=str('Lsol / AA'))
#     tacle.save_spectra('Dust', dust_spectra, resample_wl, units=str('Lsol / AA'))        
#     # tacle.save_spectra('Intrinsic', intrinsic_spectra, wl, units=str('Lsol / AA'))
#     # tacle.save_spectra('Dust', dust_spectra, wl, units=str('Lsol / AA'))
# 
#     # M_g = tacle.calculate_photometry(intrinsic_spectra, resample_wl, filter_name='SDSS_g')
#     # M_r = tacle.calculate_photometry(intrinsic_spectra, resample_wl, filter_name='SDSS_r')
#     # tacle.save_arr(M_g,'M_g','Photometry/Intrinsic/')
#     # tacle.save_arr(M_g,'M_g','Photometry/Intrinsic/')
# 
#     # M_g = tacle.calculate_photometry(dust_spectra, resample_wl, filter_name='SDSS_g')
#     # M_r = tacle.calculate_photometry(dust_spectra, resample_wl, filter_name='SDSS_r')
#     # tacle.save_arr(M_g,'M_g','Photometry/Dust/')
#     # tacle.save_arr(M_g,'M_g','Photometry/Dust/')
#     
#     # ## Add noise and save
#     # intrinsic_spectra = tacle.add_noise_flat(intrinsic_spectra, resample_wl)
#     # dust_spectra = tacle.add_noise_flat(dust_spectra, resample_wl)
# 
#     # tacle.save_spectra('Noisified Intrinsic', intrinsic_spectra, resample_wl, units=str('Lsol / AA'))
#     # tacle.save_spectra('Noisified Dust', dust_spectra, resample_wl, units=str('Lsol / AA'))
# 
# 
# if __name__ == '__main__':
#     from argparse import ArgumentParser
#     parser = ArgumentParser(description="Schwimmbad example.")
# 
#     group = parser.add_mutually_exclusive_group()
#     group.add_argument("--ncores", dest="n_cores", default=1,
#                        type=int, help="Number of processes (uses multiprocessing).")
#     group.add_argument("--mpi", dest="mpi", default=False,
#                        action="store_true", help="Run with MPI.")
#     args = parser.parse_args()
#     
#     pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)
#     print(pool)
#     main(pool)
# 
#     print("All done. Spec-tacular!")
# 


