import numpy as np
import h5py
import schwimmbad
from functools import partial
from spectacle import spectacle


def main(pool):
    tacle = spectacle.spectacle(fname='example_data.h5')
    print(tacle.package_directory)
    tacle._clear_spectra()

    shids = tacle.load_arr('Subhalos/ID')

    print('z:',tacle.redshift)
    grid = tacle.load_grid(name='fsps_neb')#, grid_directory=tacle.grid_directory)
    grid = tacle.redshift_grid(grid,tacle.redshift)
    Z = grid['metallicity']
    A = grid['age'][tacle.redshift]
    wl = grid['wavelength']

    ## Calculate weights (in parallel)
    lg = partial(tacle.weights_grid, Z=Z, A=A, resample=True)
    weights = np.array(list(pool.map(lg,shids)))
    pool.close()


    ## Dust metallicity factor
    with h5py.File(tacle.filename, 'r') as f:
        shids = f['Subhalos/ID'][:]# [:N]
        sfgmass = f["Subhalos/Star Forming Gas Mass"][:]# [:N]
        sm30 = 10**f["Subhalos/Stellar Mass 30kpc"][:]# [:N] 
        sfgmet = f["Subhalos/Stellar Metallicity 30kpc"][:]# [:N]

    tau_0 = tacle.tau_0()
    print("tau_0:",tau_0)
    Z_factor = tacle.metallicity_factor_update(sfgmet,sfgmass,sm30,gas_norm=1,metal_norm=1,tau_0=290)#tau_0)

    ## Calculate intrinsic and two-component dust attenuated spectra
    intrinsic_spectra = tacle.calc_intrinsic_spectra(weights,grid,z=tacle.redshift)
    dust_spectra = tacle.two_component_dust(weights,grid,z=tacle.redshift,
                                            tau_ism=0.33 * Z_factor, tau_cloud=0.67 * Z_factor)

    ## Rebin spectra
    resample_wl = np.loadtxt('wavelength_grid.txt')
    intrinsic_spectra = tacle.rebin_spectra(intrinsic_spectra, grid['wavelength'],resample_wl)
    dust_spectra = tacle.rebin_spectra(dust_spectra, grid['wavelength'],resample_wl)

    ## Save to HDF5
    tacle.save_spectra('Intrinsic', intrinsic_spectra, resample_wl, units=str('Lsol / AA'))
    tacle.save_spectra('Dust', dust_spectra, resample_wl, units=str('Lsol / AA'))
    tacle.save_spectra('Intrinsic', intrinsic_spectra, wl, units=str('Lsol / AA'))
    tacle.save_spectra('Dust', dust_spectra, wl, units=str('Lsol / AA'))

    ## Photometry
    M_g = tacle.calculate_photometry(intrinsic_spectra, resample_wl, filter_name='SDSS_g')
    M_r = tacle.calculate_photometry(intrinsic_spectra, resample_wl, filter_name='SDSS_r')
    tacle.save_arr(M_g,'M_g','Photometry/Intrinsic/')
    tacle.save_arr(M_g,'M_g','Photometry/Intrinsic/')

    M_g = tacle.calculate_photometry(dust_spectra, resample_wl, filter_name='SDSS_g')
    M_r = tacle.calculate_photometry(dust_spectra, resample_wl, filter_name='SDSS_r')
    tacle.save_arr(M_g,'M_g','Photometry/Dust/')
    tacle.save_arr(M_g,'M_g','Photometry/Dust/')

    ## Add noise and save
    intrinsic_spectra = tacle.add_noise_flat(intrinsic_spectra, resample_wl)
    dust_spectra = tacle.add_noise_flat(dust_spectra, resample_wl)

    tacle.save_spectra('Noisified Intrinsic', intrinsic_spectra, resample_wl, units=str('Lsol / AA'))
    tacle.save_spectra('Noisified Dust', dust_spectra, resample_wl, units=str('Lsol / AA'))
    



if __name__ == '__main__':
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

    print("All done. Spec-tacular!")



