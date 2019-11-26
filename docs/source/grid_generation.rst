Grid Generation
***************

Any SPS model can be used to generate an age-metallicity grid. We provide scripts for generating grids for the following:

* FSPS 
* BC03 (Galaxev) [see `here <http://www.bruzual.org/bc03/Original_version_2003/>`_]
* BPASS [see `here <http://bpass.auckland.ac.nz/>`_]
  
For BC03 and BPASS you need to download the original spectra from the respective home pages. 

For FSPS you can simply run the script as-is. You will need `python-FSPS <http://dfm.io/python-fsps/current/installation/>`_, astropy and numpy installed.

.. code-block:: bash

    cd grids
    python grid_fsps.py

This will generate two grids, one including the nebular contribution from young stars and one without (See Byler+17 for details on how the nebular contribution is calculated self-consistently in FSPS).

