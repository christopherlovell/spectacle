
import os
import numpy as np
import h5py
from get import get, get_sub_spectra, get_sub_url,\
        get_sub_particles, get_sub_particles_fiber

fname = 'example_data.h5'


r = get('http://www.tng-project.org/api/')
sim = get( r['simulations'][0]['url'] )
snap = get( get(sim['snapshots'])[125]['url'] ) # z = 0.1, snap 127
print(snap['number'])

temp = get('http://www.tng-project.org/api/%s'%sim['name'])
filename = get(temp['files']['pmsd'])

with h5py.File(filename, 'r') as hf:
    sh_id = hf['Snapshot_127']['SubfindID'][:]
    mstar_30 = np.log10(hf['Snapshot_127']['Mstar_30'][:] / sim['hubble'])



## ---- filter galaxies 
subhalo_ids = sh_id[(mstar_30 > 10.07) & (mstar_30 < 10.09)]
shape = len(subhalo_ids)
print("N(galaxies):", shape)


with h5py.File(fname,'a') as f:
    
    if "Star Particles" not in list(f.keys()): f.create_group("Star Particles")
    if "Subhalos" not in list(f.keys()): f.create_group("Subhalos")

    ## delete fields ##
    for field in list(f["Subhalos"].keys()):
        del f['Subhalos/%s'%field]
            
    for field in list(f["Star Particles"].keys()):
        del f['Star Particles/%s'%field]
        
        
    # set redshift
    f.attrs['redshift'] = snap['redshift']




with h5py.File(fname,'a') as f:
    
    if "ID" not in list(f["Subhalos"].keys()):
        f["Subhalos"].create_dataset('ID',(shape,),dtype='i8',maxshape=(None,),data=np.ones(shape)*-1)
    
    if "Stellar Mass" not in list(f["Subhalos"].keys()):
        f["Subhalos"].create_dataset('Stellar Mass',(shape,),maxshape=(None,))
        
    if "Gas Metallicity" not in list(f["Subhalos"].keys()):
        f["Subhalos"].create_dataset('Gas Metallicity',(shape,),maxshape=(None,))
        
    if "Stellar Metallicity" not in list(f["Subhalos"].keys()):
        f["Subhalos"].create_dataset('Stellar Metallicity',(shape,),maxshape=(None,))
        
        f["Subhalos"].create_dataset('SFR',(shape,),maxshape=(None,))
        
    if "Star Forming Gas Mass" not in list(f["Subhalos"].keys()):
        f['Subhalos'].create_dataset('Star Forming Gas Mass',(shape,),maxshape=(None,))
        
    if "Stellar Mass 30kpc" not in list(f["Subhalos"].keys()):
        f['Subhalos'].create_dataset('Stellar Mass 30kpc',(shape,),maxshape=(None,))
        
    if "Stellar Metallicity 30kpc" not in list(f["Subhalos"].keys()):
        f['Subhalos'].create_dataset('Stellar Metallicity 30kpc',(shape,),maxshape=(None,))
        
    if "Index Start" not in list(f["Subhalos"].keys()):
        f["Subhalos"].create_dataset('Index Start',(shape,),dtype='i8',maxshape=(None,))
        
    if "Index Length" not in list(f["Subhalos"].keys()):
        f["Subhalos"].create_dataset('Index Length',(shape,),dtype='i8',maxshape=(None,))
    
    if "Formation Time" not in list(f["Star Particles"].keys()):
        f["Star Particles"].create_dataset('Formation Time',(0,),maxshape=(None,))
        
    if "Initial Mass" not in list(f["Star Particles"].keys()):
        f["Star Particles"].create_dataset('Initial Mass',(0,),maxshape=(None,))
        
    if "Metallicity" not in list(f["Star Particles"].keys()):
        f["Star Particles"].create_dataset('Metallicity',(0,),maxshape=(None,))



with h5py.File(fname,'a') as f:

    pidx = len(f["Star Particles/Formation Time"][:])
    print("Initial pidx:", pidx)
    
    for i, sid in enumerate(subhalo_ids):
        
        if sid in f["Subhalos/ID"][:]:
            continue

        if (i % 100) == 0: print(round(float(i)/len(subhalo_ids) * 100,2), "%")

        out = get(get_sub_url(sid, sim=sim['name'], snap=snap['number']));
        mstar, gas_metallicity, sfr, stellar_metallicity = [out[key] for key in \
                    ['mass_stars','gasmetallicitysfr','sfr','starmetallicity']]

        particles, star_forming_gas_mass, com = \
            get_sub_particles(sid, sim=sim['name'], snap=snap['number']);
        
        f["Subhalos/ID"][i] = sid
        f["Subhalos/Stellar Mass"][i] = (mstar * 1e10) / sim['hubble']
        f["Subhalos/Gas Metallicity"][i] = gas_metallicity
        f["Subhalos/SFR"][i] = sfr
        f["Subhalos/Star Forming Gas Mass"][i] = star_forming_gas_mass
        f["Subhalos/Stellar Mass 30kpc"][i] = mstar_30[sh_id == sid][0]
        f["Subhalos/Stellar Metallicity 30kpc"][i] = np.mean(particles['Metallicity'])

        plen = len(particles['formationTime'])
        print(pidx,plen,pidx+plen)
        
        f["Subhalos/Index Start"][i] = pidx
        f["Subhalos/Index Length"][i] = plen
        
        dset = f["Star Particles/Formation Time"]
        dset.resize(size=(pidx+plen,))
        dset[pidx:] = particles['formationTime']
        
        dset = f["Star Particles/Initial Mass"]
        dset.resize(size=(pidx+plen,))
        dset[pidx:] = particles['InitialStellarMass']
        
        dset = f["Star Particles/Metallicity"]
        dset.resize(size=(pidx+plen,))
        dset[pidx:] = particles['Metallicity']
        
        pidx += plen
        
        f.flush()        




if os.path.exists("pmsd.hdf5"): os.remove("pmsd.hdf5")
if os.path.exists("temp.hdf5"): os.remove("temp.hdf5")
