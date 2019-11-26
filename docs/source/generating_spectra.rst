Generating Spectra
******************

We provide a number of example scripts in the :code:`example` directory that demonstrate how to run Spectacle to generate spectra. To get started, first download some example data from the Illustris API:

.. code:: bash

    python download_illustris.py


The main component of spectacle is the :code:`spectacle` class, which must be instantiated with the location of the hdf5 file containing the particle data:

.. code:: python

    tacle = spectacle.spectacle(fname='example_data.h5')


We can then load a grid as so:

.. code:: python

    grid = tacle.load_grid(name='fsps_neb')
    grid = tacle.redshift_grid(grid,tacle.redshift)


The most expensive part of the spectra generation is calculating the grid weights. We use the :code:`schwimmbad` pool syntax to do this flexibly in parallel (we discuss the parallelisation in more detail in FlexPara_). The :code:`partial` syntax allows us to specify additional arguments to the pool, such as the age and metallicty grid values, and whether we wish to resample. It's then simply a case of calling :code:`pool.map` and converting the output to an array, before closing the pool.

.. code:: python

    lg = partial(tacle.weights_grid, Z=grid['metallicity'], 
                 A=grid['age'][tacle.redshift], resample=True)
    weights = np.array(list(pool.map(lg,shids)))
    pool.close()


Now we have our weights we can use these to directly calculate the intrinsic spectra

.. code:: python

    intrinsic_spectra = tacle.calc_intrinsic_spectra(weights,grid,z=tacle.redshift)


There are also a number of functions available for applying a dust-screen, using the subhalo properties of each galaxy.

