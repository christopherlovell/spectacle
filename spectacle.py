import pickle as pcl
import sys
import numpy as np
import h5py
from weights import calculate_weights
import schwimmbad
from astropy.cosmology import Planck15 as cosmo
from functools import partial



def load_galaxy(shidx,h5file):
    """
    Load particle data for a single galaxy from an hdf5 file
    """
    f = h5py.File(h5file, 'r')
    midx = np.where(shidx == f['Subhalos/ID'][:])[0][0]
    
    pidx = f['Subhalos/Index Start'][midx]
    plen = f['Subhalos/Index Length'][midx]

    ftime = f['Star Particles/Formation Time'][pidx:(pidx+plen)]
    met = f['Star Particles/Metallicity'][pidx:(pidx+plen)]
    imass = f['Star Particles/Initial Mass'][pidx:(pidx+plen)]

    f.close()
    
    return (ftime,met,imass)



def load_grid(name='bc03_chab', z0=0.0):
    """
    Load SPS model
    """
    grid_directory = '/research/astro/highz/Students/Chris/sph2sed/grids'
    file_dir = '%s/intrinsic/output/%s.p'%(grid_directory,name)
   
    # if verbose: print("Loading %s model from: \n\n%s\n"%(name, file_dir))
    temp = pcl.load(open(file_dir, 'rb'))

    grid = {'name': name, 'grid': None, 'age': None, 'metallicity':None}
    grid['grid'] = temp['Spectra']
    grid['metallicity'] = temp['Metallicity']
    grid['age'] = {z0: temp['Age']}  # scale factor
    grid['lookback_time'] = {z0: cosmo.lookback_time((1. / temp['Age']) - 1).value}  # Gyr
    grid['age_mask'] = {z0: np.ones(len(temp['Age']), dtype='bool')}
    grid['wavelength'] = temp['Wavelength']

    ## Sort grids
    if grid['age'][z0][0] > grid['age'][z0][1]:
         # if verbose: print("Age array not sorted ascendingly. Sorting...\n")
         grid['age'][z0] = grid['age'][z0][::-1]
         grid['age_mask'][z0] = grid['age_mask'][z0][::-1]
         grid['lookback_time'][z0] = grid['lookback_time'][z0][::-1]
         grid['grid'] = grid['grid'][:,::-1,:] 


    if grid['metallicity'][0] > grid['metallicity'][1]:
        #  if verbose: print("Metallicity array not sorted ascendingly. Sorting...\n")
        grid['metallicity'] = grid['metallicity'][::-1]
        grid['grid'] = grid['grid'][::-1,:,:]

 
    return grid


def weights_grid(shid,h5file,grid):
    """
    Calculate SPS weights
    """
    ftime,met,imass = load_galaxy(shid,h5file)
    
    in_arr = np.array([met,ftime,imass],dtype=np.float64)

    Z = grid['metallicity']
    A = grid['age'][0.0] # TODO: update with changing redshift

    weights_temp = calculate_weights(Z, A, in_arr)
    return np.sum(weights_temp > 0)



def resample_sf():
    """
    Resample recent star formation, save to HDF5 file
    """
    return None


def intrinsic_spectra():
    """
    return intrinsic spectra
    """
    return None


def dust_attenuated_spectra():
    """
    return dust attenuated spectra
    """
    return None


def rebin_spectra():
    """
    resample spectra
    """
    return None


def add_noise():
    """
    add noise to spectra
    """
    return None



def main(pool):
    h5file = '../derivedSFH/data/full_histories_illustris_hmass.h5'
    f = h5py.File(h5file, 'r')
    shids = f['Subhalos/ID'][:10]
    f.close()

    grid = load_grid()
    
    #lg = partial(load_galaxy, h5file=h5file)
    lg = partial(weights_grid, h5file=h5file, grid=grid)

    print(list(pool.map(lg,shids)))

    pool.close()


if __name__ == '__main__':
    print("spectacle v0.1")

    from argparse import ArgumentParser
    parser = ArgumentParser(description="Schwimmbad example.")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ncores", dest="n_cores", default=1,
                       type=int, help="Number of processes (uses multiprocessing).")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")
    args = parser.parse_args()
    
    pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)
    print(pool)
    main(pool) 



# from sph2sed.model import sed
# sim = 'illustris_hmass'
# sp = sed()
# sp.grid_directory = '/research/astro/highz/Students/Chris/sph2sed/grids'
# sp.load_grid('bc03_chab')
# sp.redshift = 0.1
# sp.redshift_grid(z=sp.redshift)
# 
# from weights import calculate_weights
# 
# Z = sp.grid['metallicity']
# A = sp.grid['age'][sp.redshift]
# 


 
