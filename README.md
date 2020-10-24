# ExoSim
From [Sarkar 2016](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/9904/99043R/Exploring-the-potential-of-the-ExoSim-simulator-for-transit-spectroscopy/10.1117/12.2234216.short?SSO=1&tab=ArticleLink):

“ExoSim is a novel, generic, numerical end-to-end simulator of transit spectroscopy intended as open-access software. It permits the simulation of a time-resolved spectroscopic observation in either primary transit or secondary eclipse. The observational parameters can be adjusted, and the telescope and instrument parameters changed in a simple manner to simulate a variety of existing or proposed instruments. ExoSim is a tool to explore a variety of signal and noise issues that occur in, and might bias, transit spectroscopy observations, including the effects of the instrument systematics, correlated noise sources, and stellar variability. The simulations are fast, which allows ExoSim to be used for Monte Carlo simulations of such observations. ExoSim is versatile and has been applied to existing instruments such as the Hubble Wide Field Camera 3, as well as planned instruments, where it is being used in the study phase of the proposed ARIEL exoplanet characterization mission.”

This fork of ExoSim has simulating JWST as its goal. There are a number of features not in ExoSim that are required to accurately simulate JWST, such as:
- Curved spectral traces
- Multiple-ordered traces
- Ability to use WebbPSF PSFs
- Emulation of JWST detector readout patterns

This is a work-in-progress. See the [projects](https://github.com/davecwright3/ExoSimPublic/projects) page for current status.
