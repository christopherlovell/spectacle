import pickle as pcl
import numpy as np
import astropy.units as u
import pyphot

c = 2.9979e18  # AA s^-1

class photo:

    def _initialise_pyphot(self):
        self.filters = pyphot.get_library()

    
    def flux_frequency_units(self, L, wavelength, restframe=True):
        """
        Convert from Lsol AA^-1 -> erg s^-1 cm^-2 Hz^-1 in restframe (10 pc)
    
        Args:
            L [Lsol AA^-1]
            wavelength [AA]
        """
        if restframe:
            d_factor = 1.1964951828447575e+40  # 4 * pi * (10 pc -> cm)**2
        else:
            d_factor = 4 * np.pi * self.cosmo.luminosity_distance(self.redshift).to(u.cm).value**2

        c = 2.9979e18    # AA s^-1
        Llamb = L * 3.826e33               # erg s^-1 AA^1
        Llamb /= d_factor                  # erg s^-1 cm^-2 AA^-1
        return Llamb * (wavelength**2 / c)   # erg s^-1 cm^-2 Hz^-1 



    @staticmethod
    def photo(fnu, lamb, filt_trans, filt_lamb):
        """
        Absolute magnitude
    
        Args:
            fnu - flux [erg s^-1 cm^-2 Hz^-1]
            lamb - wavelength [AA]
            ftrans - filter transmission over flamb
            flamb - filter wavelength [AA]
        """

        nu = c / lamb
        
        filt_nu = c / filt_lamb

        if ~np.all(np.diff(filt_nu) > 0):
            filt_nu = filt_nu[::-1]
            filt_trans = filt_trans[::-1]
    
        ftrans_interp = np.interp(nu, filt_nu, filt_trans)
        a = np.trapz(fnu * ftrans_interp / nu, nu)
        b = np.trapz(ftrans_interp / nu, nu)

        mag = -2.5 * np.log10(a / b) - 48.6  # AB
    
        # AB magnitude, mean monochromatic flux
        return mag, a/b



    def calculate_photometry(self, spectra, wavelength, filter_name='SDSS_g', restframe_filter=True, redshift=None, user_filter=None, verbose=False):
        """
        Args:
            idx (int) galaxy index
            filter_name (string) name of filter in pyphot filter list
            spectra (string) spectra identifier, *assuming restframe luminosity per unit wavelength*
            wavelength (array) if None, use the self.wavelenght definition, otherwise define your own wavelength array
            verbose (bool)
        """

        if 'filters' not in self.__dict__:
            if verbose: print('Loading filters..')
            self._initialise_pyphot()

        # get pyphot filter
        if user_filter is not None:
            f = user_filter
        else:
            f = self.filters[filter_name]

        spec = spectra.copy()

        if restframe_filter:
            filt_lambda = np.array(f.wavelength.to('Angstrom'))
            spec = self.flux_frequency_units(spec, wavelength, restframe_filter)
        else:
            filt_lambda = np.array(f.wavelength.to('Angstrom')) / (1+self.redshift)
            spec = self.flux_frequency_units(spec, wavelength, restframe_filter)


        # if wavelength is None:
        #     wavelength = self.spectra[spectra]['lambda'] # AA

        write_name = "%s %s"%(filter_name, spectra)
        M = self.photo(spec, wavelength, f.transmit, filt_lambda)[0]
        return M

