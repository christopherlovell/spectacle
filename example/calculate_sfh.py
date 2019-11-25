import numpy as np
import h5py
import schwimmbad
from functools import partial
from spectacle import spectacle


def main(pool):
    tacle = spectacle.spectacle(fname='example/example_data.h5')
    tacle._clear_spectra()

    shids = tacle.load_arr('ID','Subhalos')

    ## SFR timescales (in parallel)
    print('SFR 100:')
    lg = partial(tacle.recalculate_sfr, time=0.1)
    sfr100 = np.array(list(pool.map(lg,shids)))
    
    print('SFR 10:')
    lg = partial(tacle.recalculate_sfr, time=0.01)
    sfr10 = np.array(list(pool.map(lg,shids)))
    
    print('SFR 50:')
    lg = partial(tacle.recalculate_sfr, time=0.05)
    sfr50 = np.array(list(pool.map(lg,shids)))
    
    print('SFR 500:')
    lg = partial(tacle.recalculate_sfr, time=0.5)
    sfr500 = np.array(list(pool.map(lg,shids)))
    
    print('SFR 1000:')
    lg = partial(tacle.recalculate_sfr, time=1.0)
    sfr1000 = np.array(list(pool.map(lg,shids)))
    pool.close()

    tacle.save_arr(sfr100,'SFR 100Myr','Subhalos')
    tacle.save_arr(sfr10,'SFR 10Myr','Subhalos')
    tacle.save_arr(sfr50,'SFR 50Myr','Subhalos')
    tacle.save_arr(sfr500,'SFR 500Myr','Subhalos')
    tacle.save_arr(sfr1000,'SFR 1Gyr','Subhalos')


    ## Bin histories in parallel 
    lg = partial(tacle.bin_histories, binning='log')
    sfh = np.array(list(pool.map(lg,shids)))
    pool.close()

    tacle.save_arr(sfh,'log_8','SFH')



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



