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

    def __init__(self, fname, redshift=None):
        #self.package_directory = os.path.dirname(os.path.abspath(__file__))          # location of package

        ## Use bash environment variable
        self.package_directory = os.environ["SPECTACLE_HOME"] 


        self.grid_directory = self.package_directory+'/grids' # location of SPS grids
        self.filter_directory = self.package_directory+'/filters'  # location of filters
        self.filename = fname
        self.cosmo = cosmo     # astropy cosmology
        self.age_lim = 0.1     # Young star age limit, used for resampling recent SF, Gyr
        
        try:
            with h5py.File(self.filename,'r') as f:
                self.redshift = f.attrs['redshift']
        except (IOError,KeyError):
            if redshift is None:
                print("No redshift provided! Exiting")
                raise ValueError('No redshift provided! exiting')
            else:
                self.redshift = redshift
            
            print("File doesn't exist, creating...")
            with h5py.File(self.filename,'a') as f:
                f.attrs['redshift'] = self.redshift
                print("File created!")


        self._initialise_pyphot()

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


#     def save_arr(self, arr, name, replace=True):
#         """
#         Load Subhalo array from file
#         """
#         maxshape = tuple([None] * arr.ndim)
#         
#         with h5py.File(self.filename, 'a') as f:
# 
# 
#             if name not in list(f['/%s'%group].keys()):
#                 f["/%s"%group].create_dataset(name, shape=arr.shape,
#                                              maxshape=maxshape,data=arr)
#             else:
#                 if replace:
#                     if len(arr) == len(f["/%s/%s"%(group,name)][:]):
#                         f["/%s/%s"%(group,name)][:] = arr
#                     else:
#                         raise ValueError('Error: New data shape not equal to existing shape')
#                 # raise ValueError('Key already in Subhalos group!')


    def _check_h5py(self, filename, obj_str):
        with h5py.File(filename, 'a') as h5file:
            if obj_str not in h5file:
                return False
            else:
                return True


    def save_arr(self, data, name, group, filename=None, overwrite=False):
        if filename is None:
            filename = self.filename

        name = "%s/%s"%(group,name)
        check = self._check_h5py(filename, name)

        with h5py.File(filename, 'a') as h5file:
            if check:
                if overwrite:
                    print('Overwriting data in %s'%name)
                    del h5file[name]
                    h5file[name] = data
                else:
                    raise ValueError('Dataset already exists, and `overwrite` not set')
            else:
                h5file.create_dataset(name, data=data)
       


    def delete_arr(self, name, group=''):
        """
        Delete Dataset from file
        """
        with h5py.File(self.filename, 'a') as f:

            if group not in f:
                print("Group doesn't exist...")
                return False

            if name not in list(f['/%s'%group].keys()):
                print("'%s' Dataset doesn't exist..."%name)
                return False

            try:
                del f["/%s/%s"%(group,name)]
            except KeyError:
                print("Can't find group/dataset...")
                return False

            return True




    def load_arr(self, name, fname=None):
        """
        Load Dataset array from file
        """
        if fname is None:
            fname = self.filename

        with h5py.File(fname, 'r') as f:
            
            if name not in f:
                raise ValueError("'%s' Dataset doesn't exist..."%name)

            arr = np.array(f.get(name))

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


