.. _FlexPara:

Flexible Parallelisation
************************

Thanks to :code:`schwimmbad` we have a lot of flexibility in the form of parallelisation we wish to perform. The example script contains a nice example from the `schwimmbad docs <https://schwimmbad.readthedocs.io/en/latest/examples/index.html#selecting-a-pool-with-command-line-arguments>`_ using command line arguments to select between these, shown below for completeness. The :code:`main` code block then runs agnostic to the chocie of paralleisation (threaded, MPI, etc.).

.. code:: python

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

