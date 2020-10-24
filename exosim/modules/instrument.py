from ..classes.sed import Sed
from ..classes.channel import Channel
from ..lib         import exolib

from exosim.lib.exolib import exosim_msg
import time
import numpy           as np
import quantities      as pq
import scipy.constants as spc
import scipy.interpolate

def run(opt, star, planet, zodi):
  
  exosim_msg('Run instrument model ... ')
  st = time.time()
  instrument_emission     = Sed(star.sed.wl, 
                                np.zeros(star.sed.wl.size, dtype=np.float64)* \
                                pq.W/pq.m**2/pq.um/pq.sr)
  instrument_transmission = Sed(star.sed.wl,
                                np.ones(star.sed.wl.size, dtype=np.float64))

  for op in opt.common_optics.optical_surface:
    dtmp_tr = np.loadtxt(op.transmission.replace('__path__', opt.__path__),
                         delimiter=',')
    dtmp_em = np.loadtxt(op.emissivity.replace(  '__path__', opt.__path__),
                         delimiter=',')

    tr = Sed(dtmp_tr[:,0]*pq.um, dtmp_tr[:,1]*pq.dimensionless)
    tr.rebin(opt.common.common_wl)

    em = Sed(dtmp_em[:,0]*pq.um, dtmp_em[:,1]*pq.dimensionless)
    em.rebin(opt.common.common_wl)
    
    exolib.sed_propagation(star.sed, tr)
    exolib.sed_propagation(zodi.sed, tr)
    exolib.sed_propagation(instrument_emission, tr, emissivity=em,
                           temperature=op())
    
    instrument_transmission.sed = instrument_transmission.sed*tr.sed

    
  channel = {}
  for ch in opt.channel:
    channel[ch.name] = Channel(star.sed, planet.cr,
                               zodi.sed, 
                               instrument_emission,   
                               instrument_transmission,
                               options=ch)
    
    ch_optical_surface = ch.optical_surface if isinstance(ch.optical_surface, list) else \
      [ch.optical_surface]
    for op in ch.optical_surface:
      dtmp=np.loadtxt(op.transmission.replace(
          '__path__', opt.__path__), delimiter=',')
      tr = Sed(dtmp[:,0]*pq.um, \
              dtmp[:,1]*pq.dimensionless)
      tr.rebin(opt.common.common_wl)
      em = Sed(dtmp[:,0]*pq.um, \
               dtmp[:,2]*pq.dimensionless)
      em.rebin(opt.common.common_wl)
      exolib.sed_propagation(channel[ch.name].star, tr)
      exolib.sed_propagation(channel[ch.name].zodi, tr)
      exolib.sed_propagation(channel[ch.name].emission, \
              tr, emissivity=em,temperature=op())
      channel[ch.name].transmission.sed *= tr.sed

   
    # Convert spectral signals
    dtmp=np.loadtxt(ch.qe().replace(
            '__path__', opt.__path__), delimiter=',')
    qe = Sed(dtmp[:,0]*pq.um, \
                 dtmp[:,1]*pq.dimensionless)
    
    Responsivity = qe.sed * qe.wl.rescale(pq.m)/(spc.c * spc.h * pq.m * pq.J)*pq.UnitQuantity('electron', symbol='e-')
    
    Re = scipy.interpolate.interp1d(qe.wl, Responsivity)
    
    Aeff = 0.25*np.pi*opt.common_optics.TelescopeEffectiveDiameter()**2
    Omega_pix = 2.0*np.pi*(1.0-np.cos(np.arctan(0.5/ch.wfno())))*pq.sr
    Apix = ch.detector_pixel.pixel_size()**2
    channel[ch.name].star.sed     *= Aeff             * \
      Re(channel[ch.name].star.wl)*pq.UnitQuantity('electron', 1*pq.counts, symbol='e-')/pq.J
    channel[ch.name].zodi.sed     *= Apix * Omega_pix * \
      Re(channel[ch.name].zodi.wl)*pq.UnitQuantity('electron', 1*pq.counts, symbol='e-')/pq.J
    channel[ch.name].emission.sed *= Apix * Omega_pix * \
      Re(channel[ch.name].emission.wl)*pq.UnitQuantity('electron', 1*pq.counts, symbol='e-')/pq.J
    
    ### create focal plane
    
    #1# allocate focal plane with pixel oversampling such that Nyquist sampling is done correctly 
    fpn = ch.array_geometry()
    fp  = np.zeros( (fpn*ch.osf()).astype(np.int) )

    #2# This is the current sampling interval in the focal plane.  
    fp_delta = ch.detector_pixel.pixel_size() / ch.osf()
    

    #### NEW: for multiple orders. General idea here is to create a new focal
    #### plane for each order. I'll accomplish this by stacking them horizontally
    #### and then breaking them apart and adding them back together in the noise module
    #### after the light curve model is created and applied. Added by David Wright 09/20


    if ch.type == "spectrometer":
      if hasattr(ch, "dispersion"):
        # check if we have multiple dispersion files, else make a list
        ch_dispersion = (
            ch.dispersion if isinstance(ch.dispersion, list) else [ch.dispersion]
        )

        # allocate some arrays
        fp_orig = fp.copy()
        fp = np.zeros(
            (
                (fpn[0] * ch.osf()).astype(np.int),
                (fpn[1] * ch.osf() * len(ch_dispersion)).astype(np.int),
            )
        )
        channel[ch.name].wl_solution = np.zeros(fp.shape[1])
        channel[ch.name].y_trace = np.zeros(fp.shape[1])  # same size as above

        for i, disp in enumerate(ch_dispersion):

            dtmp = np.loadtxt(
                disp.path.replace("__path__", opt.__path__), delimiter=","
            )
            wav = dtmp[..., 0]

            # translate one "focal plane" over, we'll restack later
            x_shift = i * fp_delta * fp_orig.shape[1]
            pathx = dtmp[..., 2] * pq.um + disp.val[0].rescale(pq.um) + x_shift
            pathy = dtmp[..., 3] * pq.um + disp.val[1].rescale(pq.um)
            pathint = scipy.interpolate.interp1d(
                pathx, pathy, bounds_error=False, fill_value=np.nan, kind="linear"
            )
            ld = scipy.interpolate.interp1d(
                pathx, wav, bounds_error=False, fill_value=0, kind="linear"
            )

            # translate as before
            x_pix_osr = np.arange(fp_orig.shape[1]) * fp_delta + x_shift
            y_pix_osr = pathint(x_pix_osr.rescale(pq.um)) * pq.um

            x_wav_osr = ld(x_pix_osr.rescale(pq.um)) * pq.um

            start = i * x_wav_osr.size
            stop = min(start + x_wav_osr.size, channel[ch.name].wl_solution.size)
            channel[ch.name].wl_solution[start:stop] = x_wav_osr.copy()
            channel[ch.name].y_trace[start:stop] = y_pix_osr.copy()

        # adding these so I don't have to rewrite much of the below code
        # Interesting behavior: can't *= with quantities
        channel[ch.name].y_trace = channel[ch.name].y_trace * pq.um
        channel[ch.name].wl_solution = channel[ch.name].wl_solution * pq.um
        y_pix_osr = channel[ch.name].y_trace
        x_wav_osr = channel[ch.name].wl_solution

      elif hasattr(ch, "ld"):
        ld = np.poly1d( (ch.ld()[1], ch.ld()[0]-ch.ld()[1]*ch.ld()[2]) )
        x_pix_osr = np.arange(fp.shape[1]) * fp_delta
        y_pix_osr = (np.zeros_like(x_pix_osr.magnitude) + fp.shape[0]//2) * fp_delta
        x_wav_osr = ld(x_pix_osr.rescale(pq.um))*pq.um
        channel[ch.name].wl_solution = x_wav_osr
      else:
          exolib.exosim_error("Dispersion law not defined.")


    elif ch.type == 'photometer':
      #4b# Estimate pixel and wavelength coordinates
      idx = np.where(channel[ch.name].transmission.sed > channel[ch.name].transmission.sed.max()/np.e)
      x_wav_osr = np.linspace(channel[ch.name].transmission.wl[idx].min().item(),
                              channel[ch.name].transmission.wl[idx].max().item(),
                              8 * ch.osf()) * channel[ch.name].transmission.wl.units
      x_wav_center = (channel[ch.name].transmission.wl[idx]*channel[ch.name].transmission.sed[idx]).sum() / \
        channel[ch.name].transmission.sed[idx].sum()
      
      channel[ch.name].wl_solution = np.repeat(x_wav_center, fp.shape[1])
    
    else:
      exolib.exosim_error("Channel should be either photometer or spectrometer.")
      

    d_x_wav_osr = np.zeros_like(x_wav_osr)
    idx = np.where(x_wav_osr > 0.0)

    # make sure we don't take the gradient between two traces
    # the original code would treat all nonzeros as if they were
    # next to each other, creating bright spots at the point where
    # the "fake" focal plane is joined
    if ch.type == 'spectrometer':
      if hasattr(ch, "dispersion"):
        idx=np.array(idx)
        num_disp = len(ch_dispersion)
        for i in np.arange(num_disp):
          # get a split of idx that covers one "focal plane"
          idx_split = idx[np.where( (idx < (i+1)*x_wav_osr.size/num_disp) & (idx >= i*x_wav_osr.size/num_disp))]
          d_x_wav_osr[idx_split] = np.gradient(x_wav_osr[idx_split])



    else:
      d_x_wav_osr[idx] = np.gradient(x_wav_osr[idx])

    if np.any(d_x_wav_osr < 0): d_x_wav_osr *= -1.0

    # Use WebbPSFs
    if (channel[ch.name].opt.webb_psf.use_webbpsf()):
      webb_psf_opt = channel[ch.name].opt.webb_psf
      psf = exolib.webb_Psf_Interp(webb_psf_opt.psf_file(),ch.osf().item(),x_wav_osr,\
                                   (y_pix_osr.rescale(pq.um)/ch.detector_pixel.pixel_size().rescale(pq.um) * ch.osf().item()))

    #5# Generate PSFs, one in each detector pixel along spectral axis
    else:
      psf = exolib.Psf(x_wav_osr, (y_pix_osr.rescale(pq.um)/ch.detector_pixel.pixel_size().rescale(pq.um) * ch.osf().item())\
                     , ch.wfno(), fp_delta, shape='airy')

    # where psf is nans, replace with zeros
    psf[np.isnan(psf)] = 0.0

    #6# Save results in Channel class
    channel[ch.name].fp_delta    = fp_delta
    channel[ch.name].psf         = psf
    channel[ch.name].fp          = fp
    channel[ch.name].osf         = np.int(ch.osf())
    channel[ch.name].offs        = np.int(ch.pix_offs())
    
    channel[ch.name].planet.sed  *= channel[ch.name].star.sed
    channel[ch.name].star.rebin(x_wav_osr)
    channel[ch.name].planet.rebin(x_wav_osr)
    channel[ch.name].zodi.rebin(x_wav_osr)
    channel[ch.name].emission.rebin(x_wav_osr)
    channel[ch.name].transmission.rebin(x_wav_osr)
    channel[ch.name].star.sed     *= d_x_wav_osr
    channel[ch.name].planet.sed   *= d_x_wav_osr
    channel[ch.name].zodi.sed     *= d_x_wav_osr 
    channel[ch.name].emission.sed *= d_x_wav_osr
    
   
    if ch.type == 'spectrometer':
      j0 = np.round(np.arange(fp.shape[1]) - psf.shape[1]//2).astype(np.int)

    elif ch.type == 'photometer':
      j0 = np.repeat(fp.shape[1]//2, x_wav_osr.size)
    else:
      exolib.exosim_error("Channel should be either photometer or spectrometer.")

    j1 = j0 + psf.shape[1]

    idx = np.where((j0>=0) & (j1 < fp.shape[1]) & (np.isfinite(y_pix_osr)))[0]

    for k in idx:
      i0 = np.int((y_pix_osr[k].rescale(pq.um)/ch.detector_pixel.pixel_size().rescale(pq.um)\
                 * ch.osf())//1 - psf.shape[0]//2 + channel[ch.name].offs)
      i1 = np.int(i0 + psf.shape[0])
      channel[ch.name].fp[i0:i1, j0[k]:j1[k]] += psf[...,k] * \
                                   channel[ch.name].star.sed[k]
   
    #9# Now deal with the planet
    planet_response = np.zeros(fp.shape[1])
    i0p = np.unravel_index(np.argmax(channel[ch.name].psf.sum(axis=2)), channel[ch.name].psf[...,0].shape)[0]
    for k in idx:
      planet_response[j0[k]:j1[k]] += psf[i0p,:,k] * channel[ch.name].planet.sed[k]



    #9# Allocate pixel response function
    kernel, kernel_delta = exolib.PixelResponseFunction(
        channel[ch.name].psf.shape[0:2],
        7*ch.osf(),   # NEED TO CHANGE FACTOR OF 7 
        ch.detector_pixel.pixel_size(),
        lx = ch.detector_pixel.pixel_diffusion_length())

    channel[ch.name].fp = exolib.fast_convolution(
        channel[ch.name].fp, 
        channel[ch.name].fp_delta,
        kernel, kernel_delta)
  
    ## TODO CHANGE THIS: need to convolve planet with pixel response function
    response_temp = np.zeros(fp.shape[1])
    for i,k in enumerate(idx):
      i0 = np.int((y_pix_osr[k].rescale(pq.um)/ch.detector_pixel.pixel_size().rescale(pq.um) \
                   * ch.osf().item())//1 - psf.shape[0]//2 + channel[ch.name].offs)
      i1 = np.int(i0 + psf.shape[0])
      response_temp[k] = planet_response[k]/(1e-30+fp[(i0+i1)//2,k])

    channel[ch.name].planet = Sed(channel[ch.name].wl_solution, response_temp)
   
    ## Fix units
    channel[ch.name].fp = channel[ch.name].fp*channel[ch.name].star.sed.units
    channel[ch.name].planet.sed = channel[ch.name].planet.sed*pq.dimensionless
   
    ## Deal with diffuse radiation
    if ch.type == 'spectrometer':
      channel[ch.name].zodi.sed     = scipy.convolve(channel[ch.name].zodi.sed, 
                      np.ones(np.int(ch.slit_width()*channel[ch.name].opt.osf())), 
                      'same') * channel[ch.name].zodi.sed.units
      channel[ch.name].emission.sed = scipy.convolve(channel[ch.name].emission.sed, 
                      np.ones(np.int(ch.slit_width()*channel[ch.name].opt.osf())), 
                      'same') * channel[ch.name].emission.sed.units
    elif ch.type == 'photometer':
      channel[ch.name].zodi.sed = np.repeat(channel[ch.name].zodi.sed.sum(),
                                            channel[ch.name].wl_solution.size)
      channel[ch.name].zodi.wl = channel[ch.name].wl_solution
      channel[ch.name].emission.sed = np.repeat(channel[ch.name].emission.sed.sum(),
                                                channel[ch.name].wl_solution.size)
      channel[ch.name].emission.wl = channel[ch.name].wl_solution
      
    else:
      exolib.exosim_error("Channel should be either photometer or spectrometer.")
      
  exosim_msg(' - execution time: {:.0f} msec.\n'.format((time.time()-st)*1000.0))
  return channel

  pass
