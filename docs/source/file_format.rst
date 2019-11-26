File Format
***********

The HDF5 file containing all data can have a flexible format depending on what the user wishes to achieve, but for most of the :code:`spectacle` class functionality the following layout is expected:

#. Spectra
    #. Intrinsic
    #. Dust
#. Photometry
    #. Intrinsic
        #. M_g
        #. M_r
    #. Dust
        #. M_g
        #. M_r
#. Star Particles
    #. Formation Time
    #. Initial Mass
    #. Metallicity
#. Subhalos
    #. Gas Metallicity         
    #. ID                       
    #. Index Length            
    #. Index Start             
    #. SFR                      
    #. SFR 100Myr              
    #. SFR 10Myr               
    #. SFR 1Gyr                
    #. SFR 500Myr              
    #. SFR 50Myr               
    #. Star Forming Gas Mass 
    #. Stellar Mass            
    #. Stellar Mass 30kpc 
    #. Stellar Metallicity     
    #. Stellar Metallicity 30kpc

