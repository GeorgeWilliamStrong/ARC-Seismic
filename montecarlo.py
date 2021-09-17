# >>> Import zoeppritz.py

from zoeppritz import *


# >>> Generate Random Distributions

def random_uniform(minimum, maximum, no_samples):
    """
    input: minimum value, maximum value and number of samples
    (no_samples)

    output: pseudo random uniform distribution between minimum
    and maximum values of length no_samples
    """

    distribution = np.array([random.uniform(minimum, maximum) for i in range(no_samples)])

    return distribution


# >>> Isotropic Monte Carlo Simulation

def isotropic_monte_carlo(vp1_min, vp1_max, vp2_min, vp2_max,
                          vs1_min, vs1_max, vs2_min, vs2_max,
                          p1_min, p1_max, p2_min, p2_max,
                          num_samples, config='Rpp', res=150):
    """
    input: minimum and maximum values for vp1, vp2, vs1, vs2,
    p1, p2, the number of samples (num_samples), coeff type
    (e.g. config='Rpp' for p-p reflection or 'Tps' for p-s
    transmission) and resolution (res)

    output: isotropic coefficient matrix (coeff) of size
    (no_samples, len(angles)) where each row represents a
    different sample and each column represents a different
    incidence angle, and an array which contains the coefficient
    profile computed from the mean values for each of the
    parameter distributions (mean)
    """

    # generate random uniform distribution for each parameter
    vp1 = random_uniform(vp1_min, vp1_max, num_samples)
    vp2 = random_uniform(vp2_min, vp2_max, num_samples)
    vs1 = random_uniform(vs1_min, vs1_max, num_samples)
    vs2 = random_uniform(vs2_min, vs2_max, num_samples)
    p1 = random_uniform(p1_min, p1_max, num_samples)
    p2 = random_uniform(p2_min, p2_max, num_samples)

    # generate list of angles
    angles = np.linspace(0, 90, res)

    # create index to extract correct coefficient from scattering matrix Q
    if config == 'Rpp':
        ind = (0, 0)
    elif config == 'Rps':
        ind = (1, 0)
    elif config == 'Tpp':
        ind = (2, 0)
    elif config == 'Tps':
        ind = (3, 0)
    else:
        raise Exception('config type not recognized')

    # initialise coeff matrix, where each row represents a different sample
    # and each column represents a different incidence angle
    coeff = np.zeros((num_samples, len(angles)), dtype=complex)

    # loop through each sample, then generate coefficients for each angle in
    # angles for that sample and store as a row in coeff
    for i in tqdm(range(num_samples)):
        coeff[i] = np.array([isotropic_zoeppritz(vp1[i], vp2[i], vs1[i],
                                                 vs2[i], p1[i], p2[i], j)[ind[0]][ind[1]] for j in angles])

    # calculate coeff for mean parameters
    mean = np.array([isotropic_zoeppritz(np.mean(vp1), np.mean(vp2), np.mean(vs1),
                                         np.mean(vs2), np.mean(p1), np.mean(p2), j)[ind[0]][ind[1]] for j in angles])

    return coeff, mean


# >>> Anisotropic Monte Carlo Simulation

def anisotropic_monte_carlo(vp1_min, vp1_max, vp2_min, vp2_max,
                            vs1_min, vs1_max, vs2_min, vs2_max, p1_min, p1_max,
                            p2_min, p2_max, e1_min, e1_max, e2_min, e2_max, d1_min,
                            d1_max, d2_min, d2_max, g1_min, g1_max, g2_min, g2_max,
                            a_angle, num_samples, config='Rpp', res=150, p_white=1e-7):
    """
    input: minimum and maximum values for vp1, vp2, vs1, vs2,
    p1, p2, e1, e2, d1, d2, g1, g2, the azimuth angle (a_angle),
    the number of samples (num_samples), coeff type (e.g.
    config='Rpp' for p-p reflection or 'Tps' for p-s
    transmission), resolution (res), pre-whitening (p_white)

    output: anisotropic coefficient matrix (COEFF) of size
    (no_samples, len(angles)) where each row represents a
    different sample and each column represents a different
    incidence angle, and an array which contains the coefficient
    profile computed from the mean values for each of the
    parameter distributions (mean)
    """

    # generate random uniform distribution for each parameter
    vp1 = random_uniform(vp1_min, vp1_max, num_samples)
    vp2 = random_uniform(vp2_min, vp2_max, num_samples)
    vs1 = random_uniform(vs1_min, vs1_max, num_samples)
    vs2 = random_uniform(vs2_min, vs2_max, num_samples)
    p1 = random_uniform(p1_min, p1_max, num_samples)
    p2 = random_uniform(p2_min, p2_max, num_samples)
    e1 = random_uniform(e1_min, e1_max, num_samples)
    e2 = random_uniform(e2_min, e2_max, num_samples)
    d1 = random_uniform(d1_min, d1_max, num_samples)
    d2 = random_uniform(d2_min, d2_max, num_samples)
    g1 = random_uniform(g1_min, g1_max, num_samples)
    g2 = random_uniform(g2_min, g2_max, num_samples)

    # create elastic tensors from parameter distributions
    c1 = np.zeros((num_samples, 6, 6))
    c2 = np.zeros((num_samples, 6, 6))
    for i in range(num_samples):
        c1[i] = thomsen_c(vp1[i], vs1[i], p1[i], e1[i], d1[i], g1[i])
        c2[i] = thomsen_c(vp2[i], vs2[i], p2[i], e2[i], d2[i], g2[i])

    # create elastic tensors from mean parameters
    c1_mean = thomsen_c(np.mean(vp1), np.mean(vs1), np.mean(p1),
                        np.mean(e1), np.mean(d1), np.mean(g1))
    c2_mean = thomsen_c(np.mean(vp2), np.mean(vs2), np.mean(p2),
                        np.mean(e2), np.mean(d2), np.mean(g2))

    # generate list of angles
    angles = np.linspace(0, 90, res)

    # create index to extract correct coefficient from scattering matrix sm
    if config == 'Rpp':
        ind = (0, 0, 0)
    elif config == 'Rps':
        ind = (0, 1, 0)
    elif config == 'Rpt':
        ind = (0, 2, 0)
    elif config == 'Tpp':
        ind = (1, 0, 0)
    elif config == 'Tps':
        ind = (1, 1, 0)
    elif config == 'Tpt':
        ind = (1, 2, 0)
    else:
        raise Exception('config type not recognized')

    # initialise coeff matrix, where each row represents a different sample
    # and each column represents a different incidence angle
    coeff = np.zeros((num_samples, len(angles)), dtype=complex)

    # loop through each sample, then generate coefficients for each angle in
    # angles for that sample and store as a row in coeff
    for i in tqdm(range(num_samples)):
        coeff[i] = np.array([anisotropic_zoeppritz(c1[i], c2[i], p1[i],
                                                   p2[i], j, a_angle, p_white)[ind[0]][ind[1]][ind[2]] for j in angles])

    # calculate coeff for mean parameters
    mean = np.array([anisotropic_zoeppritz(c1_mean, c2_mean, np.mean(p1),
                                           np.mean(p2), j, a_angle, p_white)[ind[0]][ind[1]][ind[2]] for j in angles])

    return coeff, mean


# >>> Monte Carlo Plot

def monte_carlo_plot(coeff, mean, tol=0.5, pdf=False, opacity=0.1):
    """
    input: coefficient matrix (coeff), mean array (mean), tolerance
    for extracting coefficients at correct angles for the probability
    density function (tol), boolean argument to specify whether to
    plot a probability density function or not (pdf), opacity of each
    plotted simulation profile (opacity)

    output: plots monte carlo simulation results and pdf if specified
    """

    # initialise figure
    fig = plt.figure(figsize=(13, 6))
    gs = GridSpec(20, 20)
    gs.update(wspace=0.4)
    ax1 = fig.add_subplot(gs[0:20, 0:17])

    # initialise probability density function plot if specified
    if pdf == True:
        ax2 = fig.add_subplot(gs[0:20, 17:20])

    # extract angles array from coefficient matrix
    angles = np.linspace(0, 90, np.shape(coeff)[1])

    # create array to store values for 3 different incidence angles
    angle_values = np.zeros((3, len(coeff)))

    # create magnitude matrix from coefficient matrix
    mag = abs(coeff)

    # loop through magnitude matrix and plot each sample
    for i in range(len(mag)):
        ax1.plot(angles, mag[i], linewidth=0.5, color='C0', alpha=opacity)

        # extract values for angles of 15 45 and 75 degrees
        if pdf == True:
            for j in range(len(angles)):
                if angles[j] < 15. + tol and angles[j] > 15. - tol:
                    angle_values[0][i] = mag[i][j]
                elif angles[j] < 45. + tol and angles[j] > 45. - tol:
                    angle_values[1][i] = mag[i][j]
                elif angles[j] < 75. + tol and angles[j] > 75. - tol:
                    angle_values[2][i] = mag[i][j]

    # plot mean result
    ax1.plot(angles, abs(mean), linewidth=1.5, color='C3', alpha=1)

    # tidy up plot and add labels
    ax1.set_xlim([0, 90])
    ax1.set_ylim([0, 1])
    ax1.grid()
    ax1.set_xlabel('Angle ($^\circ$)')
    ax1.set_ylabel('Magnitude')

    if pdf == True:
        # use gaussian kernel density estimates to generate probability
        # density functions for each angle
        xx = np.linspace(0, 1, 500)  # array to project KDE on to
        kde0 = stats.gaussian_kde(angle_values[0])
        ax2.plot(kde0(xx), xx, label='15$^\circ$', color='C6', alpha=1)

        kde1 = stats.gaussian_kde(angle_values[1])
        ax2.plot(kde1(xx), xx, label='45$^\circ$', color='dimgray', alpha=1)

        kde2 = stats.gaussian_kde(angle_values[2])
        ax2.plot(kde2(xx), xx, label='75$^\circ$', color='teal', alpha=1)

        # tidy up probability density function plot
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax2.set_ylim([0, 1])
        ax2.xaxis.set_major_locator(MaxNLocator(4, integer=True))
        ax2.grid()
        ax2.legend(loc='upper right')
        ax2.set_xlabel('P.D.F')

    # plt.savefig('monte_carlo.png', dpi=1000) # uncomment to save figure as image
    plt.show()
