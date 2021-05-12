import pytest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ti


def test_focalplane_signal():
    """Test basic signal produced.

    The basic signal produced in the instrument.py module is tested here.
    To generate the data with which this test is run, go into instrument.py
    and set `test_fp = True`. You'll also find this data pre-generated for both
    a flat trace and normal trace in ./references. If you would like to use that
    data to re-run these tests, rename the files to match what the test expects
    or give the `np.load()` below the files you want to use.

    """

    sim = np.load("./reference/sim_image_normal.npy")
    exact = np.load("./reference/exact_image_normal.npy")
    wl = np.load("./reference/xwav_normal.npy")
    #if(sim[-1]==0):
    #    exact[-1]=0
    i = ~((sim<=0) | (np.isnan(sim) | (exact<=0)))
    # plots
    fig1 = plt.figure(1)

    #Plot Data
    #i = np.where((exact != 0) & (sim!=0))
    #sim = sim[i]
    #exact = exact[i]
    frame1=fig1.add_axes((.13,.3,.8,.6))
    #xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
    plt.plot(wl[i],sim[i],'.c',label="Simulation")
    plt.plot(wl[i],exact[i],'-k',label="Prediction")
    plt.legend(loc='best')
    frame1.set_xticklabels([]) #Remove x-tic labels for the first frame
    frame1.set_ylabel("Signal on Detector")
    plt.grid()

    #Residual plot

    err = (sim[i]-exact[i]) / exact[i] * 100.
    frame2=fig1.add_axes((.13,.1,.8,.2))
    frame2.set_ylabel("percent error")
    frame2.set_xlabel("Wavlength (um)")
    frame2.yaxis.set_major_formatter(ti.FormatStrFormatter('%.0e'))
    plt.plot(wl[i][::3],err[::3],'.r')
    plt.grid()
    plt.savefig("./figs/sim-predict-compare-normal.pdf")
    plt.clf()
    plt.plot(err)
    plt.savefig("./figs/err-normal.pdf")

    assert np.allclose(exact[i], sim[i], rtol=0.015)
