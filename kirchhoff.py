# >>> Import zoeppritz.py

from zoeppritz import *


# >>> Ricker Wavelet

def ricker(f, t):
    """
    input: frequency in HZ (f), time array (t)

    output: ricker wavelet (x)
    """

    x = (1. - 2. * (np.pi ** 2) * (f ** 2) * (t ** 2)) * np.exp(-(np.pi ** 2) * (f ** 2) * (t ** 2))

    return x


def ricker_dt(f, t):
    """
    input: frequency in hz (f), time array (t)

    output: derivative of ricker wavelet w.r.t time
    """

    x = 2. * (np.pi ** 2) * (f ** 2) * t * np.exp(-(np.pi ** 2) * (f ** 2) * (t ** 2)) * (2. * (np.pi ** 2) *
                                                                                          (f ** 2) * (t ** 2) - 3.)

    return x


# >>> Proximity Function

def find_nearest(array, value):
    """
    input: numpy array of values (array), specified value (value)

    ouput: index of closest term in array to the specified value (idx)
    """

    idx = (np.abs(array - value)).argmin()

    return idx


# >>> Isotropic Kirchhoff Integral

def isotropic_kirchhoff_integral(vp1, vp2, vs1, vs2, p1, p2, d,
                                 l, w, ds, tmin, tmax, f, dt=5e-4):
    """
    input: upper and lower p-wave velocities (vp1, vp2), s-wave
    velocities (vs1, vs2), densities (p1, p2), interface depth (d),
    reciever offset (l), width (w), surface grid-spacing (ds), minimum
    time (tmin), maximum time (tmax), frequency (f), time increment (dt)

    output: single synthetic seismogram (trace)
    """

    width = np.arange(-w / 2., (w / 2.) + 1, ds)  # width array
    length = np.arange(0, l + 1, ds)  # length array
    time = np.arange(tmin, tmax, dt)  # time window

    # initialise matrix to store times and kirchhoff integral values
    k_values = np.zeros((len(time), 2), dtype=complex)
    for i in range(len(time)):
        k_values[i][0] = time[i]

    # loop over each surface gridpoint between source and reciever
    for i in range(len(length)):
        for j in range(len(width)):

            # calculate geometry
            r0 = np.sqrt(length[i] ** 2 + width[j] ** 2 + d ** 2)  # source to surface
            r = np.sqrt((l - length[i]) ** 2 + width[j] ** 2 + d ** 2)  # surface to reciever
            theta0 = np.arctan(float(length[i]) / float(d))  # incidence angle
            theta = np.arctan(float(l - length[i]) / float(d))  # reflection angle

            # calculate the reflection coefficient
            r_theta0 = isotropic_zoeppritz(vp1, vp2, vs1, vs2, p1, p2, np.degrees(theta0))[0][0]

            # calculate the kirchhoff integral value
            kirchhoff = ((r_theta0 * (cos(theta0) + cos(theta)) * (ds ** 2)) / (4. * np.pi * vp1 * r0 * r)) / dt

            # find the closest time in J to the corresponding ray
            index = find_nearest(k_values[:, 0], ((r0 + r) / vp1))

            # assign kirchhoff integral value to closest time
            if k_values[index][1] == 0.0:
                k_values[index][1] = kirchhoff
            else:
                k_values[index][1] = k_values[index][1] + kirchhoff

    # use the time derivative of the ricker wavelet as it is a reflection
    wavelet = ricker_dt(f, k_values[:, 0].real)

    # convolve the wavelet with the kirchhoff integral values
    convolution = np.convolve(wavelet, k_values[:, 1])[::2]

    # hilbert transform the imaginary components to account for phase shifts
    trace = convolution.real + hilbert(convolution.imag)

    return trace


# >>> Anisotropic Kirchhoff Integral

def anisotropic_kirchhoff_integral(c1, c2, p1, p2, a_angle, d,
                                   l, w, ds, tmin, tmax, f, vp1, dt=5e-4, p_white=1e-7):
    """
    input: 6x6 upper and lower elastic tensors (c1, c2), upper and
    lower densities (p1, p2), azimuth angle (a_angle), interface depth (d),
    reciever offset (l), width (w), surface grid-spacing (ds), minimum
    time (tmin), maximum time (tmax), frequency (f), upper p-wave velocity (vp1),
    time increment (dt), pre-whitening (p_white)

    output: single synthetic seismogram (trace)
    """

    width = np.arange(-w / 2., (w / 2.) + 1, ds)  # width array
    length = np.arange(0, l + 1, ds)  # length array
    time = np.arange(tmin, tmax, dt)  # time window

    # initialise matrix to store times and kirchhoff integral values
    k_values = np.zeros((len(time), 2), dtype=complex)
    for i in range(len(time)):
        k_values[i][0] = time[i]

    # loop over each surface gridpoint between source and reciever
    for i in range(len(length)):
        for j in range(len(width)):

            # calculate geometry
            r0 = np.sqrt(length[i] ** 2 + width[j] ** 2 + d ** 2)  # source to surface
            r = np.sqrt((l - length[i]) ** 2 + width[j] ** 2 + d ** 2)  # surface to reciever
            theta0 = np.arctan(float(length[i]) / float(d))  # incidence angle
            theta = np.arctan(float(l - length[i]) / float(d))  # reflection angle

            # calculate the reflection coefficient
            r_theta0 = anisotropic_zoeppritz(c1, c2, p1, p2,
                                             np.degrees(theta0), a_angle, p_white)[0][0][0]

            # calculate the kirchhoff integral value
            kirchhoff = ((r_theta0 * (cos(theta0) + cos(theta)) * (ds ** 2)) / (4. * np.pi * vp1 * r0 * r)) / dt

            # find the closest time in J to the corresponding ray
            index = find_nearest(k_values[:, 0], ((r0 + r) / vp1))

            # assign kirchhoff integral value to closest time
            if k_values[index][1] == 0.0:
                k_values[index][1] = kirchhoff
            else:
                k_values[index][1] = k_values[index][1] + kirchhoff

    # use the time derivative of the ricker wavelet as it is a reflection
    wavelet = ricker_dt(f, k_values[:, 0].real)

    # convolve the wavelet with the kirchhoff integral values
    convolution = np.convolve(wavelet, k_values[:, 1])[::2]

    # hilbert transform the imaginary components to account for phase shifts
    trace = convolution.real + hilbert(convolution.imag)

    return trace


# >>> Generate Isotropic Synthetic Data

def isotropic_synthetic(rec_min, rec_max, drec,
                        vp1, vp2, vs1, vs2, p1, p2, d, w, f, ds=20.,
                        tmin=-1, tmax=5, dt=5e-4):
    """
    input: minumum reciever distance (rec_min), maximum reciever distance
    (rec_max), reciever spacing (drec), upper and lower p-wave velocities
    (vp1, vp2), s-wave velocities (vs1, vs2), densities (p1, p2),
    interface depth (d), reciever offset (l), width (w), frequency (f), 
    surface grid-spacing (ds), minimum time (tmin), maximum time (tmax), 
    time increment (dt)

    output: array of traces (traces), time array (T)
    """

    rec = np.arange(rec_min, rec_max + 1, drec)  # initialise reciever geometry
    t = np.arange(tmin, tmax, dt)  # time array
    traces = np.zeros((len(rec), int((tmax - tmin) / dt)))  # traces array

    # loop through recievers, calculating corresponding traces
    for i in tqdm(range(len(rec))):
        traces[i] = isotropic_kirchhoff_integral(vp1, vp2, vs1, vs2, p1,
                                                 p2, d, rec[i], w, ds, tmin, tmax, f, dt)

    return traces, t


# >>> Generate Anisotropic Synthetic Data

def anisotropic_synthetic(rec_min, rec_max, drec,
                          c1, c2, p1, p2, d, w, f,
                          vp1, a_angle=0., ds=20., tmin=-1, tmax=5, dt=5e-4, p_white=1e-7):
    """
    input: minumum reciever distance (rec_min), maximum reciever distance
    (rec_max), reciever spacing (drec), 6x6 upper and lower elastic tensors
    (C1, C2), upper and lower densities (p1, p2), interface depth (d), 
    reciever offset (l), width (w), frequency (f), upper p-wave velocity (vp1), 
    azimuth angle (a_angle), surface grid-spacing (ds), minimum time (tmin), 
    maximum time (tmax), time increment (dt), pre-whitening (p_white)

    output: array of traces (traces), time array (T)
    """

    rec = np.arange(rec_min, rec_max + 1, drec)  # initialise reciever geometry
    t = np.arange(tmin, tmax, dt)  # time array
    traces = np.zeros((len(rec), int((tmax - tmin) / dt)))  # traces array

    # loop through recievers, calculating corresponding traces
    for i in tqdm(range(len(rec))):
        traces[i] = anisotropic_kirchhoff_integral(c1, c2, p1, p2,
                                                   a_angle, d, rec[i], w, ds, tmin, tmax, f, vp1, dt, p_white)

    return traces, t


# >>> Plot Synthetic Data

def plot_synthetic(traces, time, scale_fac=1, ymin=0, ymax=1.5):
    """
    input: array of traces (traces), corresponding time array (time),
    scale factor (scale_fac), y axis minimum (ymin) and maximum (ymax)

    output: plot of the synthetic traces
    """

    fig, ax1 = plt.subplots(figsize=(13, 7))

    for i in range(len(traces)):
        ax1.plot((scale_fac * traces[i]) + i, time, 'k', linewidth=1.)
        ax1.fill_betweenx(time, i, (scale_fac * traces[i]) + i,
                          where=(((scale_fac * traces[i]) + i) > i), color='k')

    ax1.set_xticks([])
    plt.xlim((-1., len(traces)))
    plt.ylim((ymin, ymax))
    ax1.invert_yaxis()
    plt.grid()
    ax1.set_ylabel('Time (s)')
    # plt.savefig('synthetic.png', dpi=1000) # uncomment to save figure as image
    plt.show()
