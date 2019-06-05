import pickle as pcl
import sys
import numpy as np
import h5py
from weights import calculate_weights
import schwimmbad
import astropy.units as u
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
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


def redshift_grid(grid, z, z0=0.0):
    """
    Redshift grid ages, return new grid
    Args:
        z (float) redshift
    """
 
    if z == z0:
        print("No need to initialise new grid, z = z0")
        return grid
    else:
        observed_lookback_time = cosmo.lookback_time(z).value
        # if verbose: print("Observed lookback time: %.2f"%observed_lookback_time)
 
        # redshift of age grid values
        age_grid_z = [z_at_value(cosmo.scale_factor, a) for a in grid['age'][z0]]
        # convert to lookback time
        age_grid_lookback = np.array([cosmo.lookback_time(z).value for z in age_grid_z])
        # add observed lookback time
        age_grid_lookback += observed_lookback_time
 
        # truncate age grid by age of universe
        age_mask = age_grid_lookback < cosmo.age(0).value
        age_mask = age_mask & ~(np.isclose(age_grid_lookback, cosmo.age(0).value))
        age_grid_lookback = age_grid_lookback[age_mask]
 
        # convert new lookback times to redshift
        age_grid_z = [z_at_value(cosmo.lookback_time, t * u.Gyr) for t in age_grid_lookback]
        # convert redshift to scale factor
        age_grid = cosmo.scale_factor(age_grid_z)
 
        grid['age'][z] = age_grid
        grid['lookback_time'][z] = age_grid_lookback - observed_lookback_time
        grid['age_mask'][z] = age_mask

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
    return weights_temp



def resample_sf():
    """
    Resample recent star formation, save to HDF5 file
    """
    return None


def calc_intrinsic_spectra(weights,grid,z=0.0):
    """
    return intrinsic spectra
    """
    grid = grid['grid'][:,grid['age_mask'][z],:]
    return np.einsum('ijk,jkl->il',weights,grid)#,optimize=True)



def two_component_dust(weights, grid, z=0.0, lambda_nu=5500, tau_ism=0.33, tau_cloud=0.67, tdisp=1e-2, gamma=0.7, gamma_cloud=0.7):
    
    
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

    # self.spectra['Dust %s'%name] = {'grid_name': self.grid['name'],
    #                                 'lambda': self.grid['wavelength'],
    #                                 'units': 'Lsol / AA', 'scaler': None}

    # return dust_spectra, weights



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


def save_spectra(h5file, name, spectra):

    f = h5py.File(h5file, 'a')
    
    if "Spectra" not in list(f.keys()): f.create_group("Spectra") 

    s1 = len(spectra)
    s2 = spectra.shape[1]

    if name not in list(f["Spectra"].keys()):
        f["Spectra"].create_dataset(name,spectra.shape,maxshape=(None,s2),data=spectra)

    f.close()



def main(pool):
    h5file = '../derivedSFH/data/full_histories_illustris_hmass.h5'
    f = h5py.File(h5file, 'r')
    shids = f['Subhalos/ID'][:30]
    f.close()
    z = 0.1

    grid = load_grid()
    grid = redshift_grid(grid,z)
 
    lg = partial(weights_grid, h5file=h5file, grid=grid)
    weights = np.array(list(pool.map(lg,shids)))
    pool.close()

    intrinsic_spectra = calc_intrinsic_spectra(weights,grid)
    dust_spectra = two_component_dust(weights,grid)
    
    print(intrinsic_spectra.shape)
    print(dust_spectra.shape)
   
    save_spectra(h5file,'Intrinsic',intrinsic_spectra)
    save_spectra(h5file,'Dust',intrinsic_spectra)

    


if __name__ == '__main__':
    # print("spectacle v0.1")

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


