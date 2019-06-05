import numpy as np
import matplotlib.pyplot as plt
import spectacle

tacle = spectacle.spectacle()

spectra, wavelength = tacle.load_spectra(name='Intrinsic')

print(spectra.shape)


# [plt.semilogx(wavelength, s, alpha=0.1,color='black') for s in spectra]
# plt.semilogx(wavelength, np.median(spectra, axis=0)) #alpha=0.6)

[plt.loglog(wavelength, s, alpha=0.1,color='black') for s in spectra]
plt.loglog(wavelength, np.median(spectra, axis=0)) #alpha=0.6)

plt.show()
