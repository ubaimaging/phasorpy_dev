import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.widgets import Cursor
import hsitools
from matplotlib.patches import Ellipse

# The following two functions define the phasor figure: circle and cluster
def phasor_circle(ax):
    """
        Built the figure inner and outer circle and the 45 degrees lines in the plot
    :param ax: axis where to plot the phasor circle.
    :return: the axis with the added circle.
    """
    x1 = np.linspace(start=-1, stop=1, num=500)
    yp1 = lambda x1: np.sqrt(1 - x1 ** 2)
    yn1 = lambda x1: -np.sqrt(1 - x1 ** 2)
    x2 = np.linspace(start=-0.5, stop=0.5, num=500)
    yp2 = lambda x2: np.sqrt(0.5 ** 2 - x2 ** 2)
    yn2 = lambda x2: -np.sqrt(0.5 ** 2 - x2 ** 2)
    x3 = np.linspace(start=-1, stop=1, num=30)
    x4 = np.linspace(start=-0.7, stop=0.7, num=30)
    ax.plot(x1, list(map(yp1, x1)), color='darkgoldenrod')
    ax.plot(x1, list(map(yn1, x1)), color='darkgoldenrod')
    ax.plot(x2, list(map(yp2, x2)), color='darkgoldenrod')
    ax.plot(x2, list(map(yn2, x2)), color='darkgoldenrod')
    ax.scatter(x3, [0] * len(x3), marker='_', color='darkgoldenrod')
    ax.scatter([0] * len(x3), x3, marker='|', color='darkgoldenrod')
    ax.scatter(x4, x4, marker='_', color='darkgoldenrod')
    ax.scatter(x4, -x4, marker='_', color='darkgoldenrod')
    ax.annotate('0º', (1, 0), color='darkgoldenrod')
    ax.annotate('180º', (-1, 0), color='darkgoldenrod')
    ax.annotate('90º', (0, 1), color='darkgoldenrod')
    ax.annotate('270º', (0, -1), color='darkgoldenrod')
    ax.annotate('0.5', (0.42, 0.28), color='darkgoldenrod')
    ax.annotate('1', (0.8, 0.65), color='darkgoldenrod')
    return ax

def plot_colormap(ax, cmap_range, colormap, zorder):
    # Create a grid for the colormap background
    X, Y = np.meshgrid(np.linspace(-1, 1, 500), np.linspace(-1, 1, 500))
    distances = np.sqrt(X**2 + Y**2)
    angles = np.arctan2(Y, X) * 180 / np.pi  # Convert radians to degrees
    colormap_vals = (angles - cmap_range[0]) / (cmap_range[1] - cmap_range[0])
    
    # Mask the angles array to show only angles within the specified cmap_range
    mask = ((angles >= cmap_range[0]) & (angles <= cmap_range[1])) & (distances <= 1)
    colormap_vals[~mask] = np.nan  # Set values outside circle to NaN

    # Plot the colormap background using imshow
    im = ax.imshow(colormap(colormap_vals), extent=[-1, 1, -1, 1], aspect='auto', origin='lower', zorder=zorder)
    
    # Set the axis limits to show the full circle
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])


def phasor_semicircle(ax, colormap, cmap_range=(0, 180)):
    """
    Build the figure with the upper half of a semicircle and a colormap background.
    :param ax: axis where to plot the upper semicircle and colormap.
    :param colormap: a matplotlib colormap to use for coloring the background.
    :param cmap_range: the range of values to map to the colormap, default is (0, 180) degrees.
    :return: the axis with the added upper semicircle and colormap background.
    """
    # Adjust the axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    theta = np.linspace(start=0, stop=np.pi, num=500)
    x = 0.5 * (np.cos(theta) + 1)  # Scale and shift x to range from 0 to 1
    y = 0.5 * np.sin(theta)  # Scale y to range from -0.5 to 0.5

    # Create a grid for the colormap background centered at (0.5, 0)
    X, Y = np.meshgrid(np.linspace(0, 1, 500), np.linspace(0, 0.5, 500))

    # Calculate distances using the ellipse equation
    distances = np.sqrt((X - 0.5) ** 2 + Y ** 2)

    # Calculate angles
    angles = np.arctan2(Y, X - 0.5) * 180 / np.pi

    # Calculate colormap_vals based on whether points are inside or outside the semicircle
    colormap_vals = np.zeros_like(angles, dtype=float)
    mask = (distances <= 0.5) & (angles >= cmap_range[0]) & (angles <= cmap_range[1])
    colormap_vals[mask] = (angles[mask] - cmap_range[0]) / (cmap_range[1] - cmap_range[0])
    colormap_vals[~mask] = np.nan
    # Plot the colormap background using imshow
    im = ax.imshow(colormap(colormap_vals), extent=[0, 1, 0, 0.5], aspect='auto', origin='lower', alpha = 0.6)
    ax.plot(x, y, color='black', linewidth = 0.5)
    ax.set_xlabel('g')
    ax.set_ylabel('s')

    # Add colorbar
    cbar = plt.cm.ScalarMappable(cmap=colormap)
    cbar.set_clim(vmin=cmap_range[0], vmax=cmap_range[1])
    # plt.colorbar(cbar)
    return ax


def phasor_circle_cmap(ax, colormap, cmap_range=(0, 360)):
    """
    Build the figure with a circle and a colormap background.
    :param ax: axis where to plot the circle and colormap.
    :param colormap: a matplotlib colormap to use for coloring the background.
    :param cmap_range: the range of values to map to the colormap, default is (0, 360) degrees.
    :return: the axis with the added circle and colormap background.
    """
    theta = np.linspace(start=0, stop=2*np.pi, num=500)
    x = np.cos(theta)
    y = np.sin(theta)

    # Create a grid for the colormap background
    X, Y = np.meshgrid(np.linspace(-1, 1, 500), np.linspace(-1, 1, 500))
    distances = np.sqrt(X**2 + Y**2)
    angles = np.arctan2(Y, X) * 180 / np.pi  # Convert radians to degrees
    # Create a mask to set values outside the semicircle trace to NaN
    mask = (distances > 1) | (angles < cmap_range[0]) | (angles > cmap_range[1])
    colormap_vals = (angles - cmap_range[0]) / (cmap_range[1] - cmap_range[0])
    colormap_vals[mask] = np.nan  # Set values outside semicircle to NaN

    # Plot the colormap background using imshow
    im = ax.imshow(colormap(colormap_vals), extent=[-1, 1, -1, 1], aspect='auto', origin='lower')

    ax.plot(x, y, color='darkgoldenrod')
    ax.scatter(0, 0, color='darkgoldenrod')  # Center point
    ax.annotate('0°', (0.02, -0.05), color='darkgoldenrod')
    ax.annotate('90°', (-0.05, 0.95), color='darkgoldenrod')
    ax.annotate('180°', (-1.1, -0.05), color='darkgoldenrod')
    ax.annotate('270°', (-0.05, -1.1), color='darkgoldenrod')
    ax.set_aspect('equal', adjustable='datalim')  # Equal aspect ratio for x and y axes

    # Add colorbar
    cbar = plt.cm.ScalarMappable(cmap=colormap)
    cbar.set_clim(vmin=cmap_range[0], vmax=cmap_range[1])
    # cbar.set_label('Phase')
    plt.colorbar(cbar)
    return ax

def phasor_figure(x, y, circle_plot=False):
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.suptitle('Phasor')
    ax.hist2d(x, y, bins=256, cmap="RdYlGn_r", norm=colors.LogNorm(), range=[[-1, 1], [-1, 1]])
    if circle_plot:
        phasor_circle(ax)
    return fig


# The following 2 functions are a set of interactive functions to plot and performe phasor analysis
def interactive1(dc, g, s, Ro, nbit, histeq=True, ncomp=5, filt=False, nfilt=0, spectrums=False,
                 hsi_stack=None, lamd=None):
    """
        This function plot the avg image, its histogram, the phasors and the rbg pseudocolor image.
    To get the phasor the user must pick an intensity cut umbral in the histogram in order to plot the phasor.
    To get the rgb pseudocolor image you must pick three circle in the phasor plot.
    :param nbit: bits of the image.
    :param dc: average intensity image. ndarray
    :param g: image. ndarray. Contains the real coordinate G of the phasor
    :param s: image. ndarray. Contains the imaginary coordinate S of the phasor
    :param Ro: radius of the circle to select pixels in the phasor

    :param lamd: Lamba array containing the wavelenght. numpy array. Optional
    :param hsi_stack: HSI stack to plot the spectrums of each circle regions.
    :param spectrums: set True to plot the average spectrum of each circle. Optional
    :param nfilt: amount of times to filt G and S images. Optional
    :param filt: Apply median filter to G and S images, before the dc threshold. Optional
    :param ncomp: number of cursors to be used in the phasor, and the pseudocolor image. Default 5.
    :param histeq: equalize histogram used in dc image for a better representation.
            Its only applies for dc when plotting it. Optional

    :return: fig: figure contains the avg, histogram, phasor and pseudocolor image.
    """
    if histeq:
        from skimage.exposure import equalize_adapthist
        auxdc = equalize_adapthist(dc / dc.max())
    else:
        auxdc = dc
    if filt:
        from skimage.filters import median
        for i in range(nfilt):
            g = median(g)
            s = median(s)
    nbit = 2 ** nbit

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.imshow(auxdc, cmap='gray')
    ax1.axis('off')
    ax1.set_title('Average intensity image')
    ax2.hist(dc.flatten(), bins=nbit, range=(0, nbit))
    ax2.set_yscale("log")
    ax2.set_title('Average intensity image histogram')
    cursor = Cursor(ax2, horizOn=False, vertOn=True, color='darkgoldenrod')
    ic = plt.ginput(1, timeout=0)
    ic = int(ic[0][0])
    x, y = hsitools.histogram_thresholding(dc, g, s, ic)

    figp, ax3 = plt.subplots(1, 1, figsize=(10, 7))
    phasor_circle(ax3)
    phasorbar = ax3.hist2d(x, y, bins=256, cmap="RdYlGn_r", norm=colors.LogNorm(),
                           range=[[-1, 1], [-1, 1]])
    ax3.set_title('Phasor', pad=20)
    plt.sca(ax3)
    plt.xticks([-1, 0, 1], ['-1', '0', '1'])
    plt.yticks([-1, 0, 1], ['-1', '0', '1'])
    fig.colorbar(phasorbar[3], ax=ax3)
    center = plt.ginput(ncomp, timeout=0)  # get the circle centers
    ccolor = ['darkviolet', 'blue', 'green', 'yellow', 'red']
    for i in range(ncomp):
        circle = plt.Circle((center[i][0], center[i][1]), Ro, color=ccolor[i], fill=False)
        ax3.add_patch(circle)

    g = np.where(dc > ic, g, dc * np.nan)
    s = np.where(dc > ic, s, dc * np.nan)
    rgba = hsitools.pseudocolor_image(dc, g, s, center, Ro, ncomp=ncomp)
    fig2, ax4 = plt.subplots(1, 1, figsize=(8, 8))
    ax4.imshow(rgba)
    ax4.set_title('Pseudocolor image')
    ax4.axis('off')

    if spectrums:
        spect = hsitools.avg_spectrum(hsi_stack, g, s, ncomp, Ro, center)
        plt.figure(figsize=(12, 6))
        for i in range(ncomp):
            if lamd.any():
                plt.plot(lamd, spect[i], ccolor[i])
            else:
                plt.plot(spect[i], ccolor[i])
        plt.grid()
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Normalize intensity')
        plt.title('Average Components Spectrums')
    plt.show()
    return fig


def interactive2(dc, g, s, nbit, phase, phint, modulation, mdint, histeq=True, filt=False, nfilt=0):
    """
        This function plot the avg image, its histogram, the phasors and the rbg pseudocolor image.
    To get the phasor the user must pick an intensity cut umbral in the histogram in order to plot
    the phasor. To get the rgb pseudocolor image you must pick three circle in the phasor plot.
    :param phint:
    :param mdint:
    :param modulation:
    :param phase:
    :param nfilt: amount of times to filt G and S images.
    :param filt: Apply median filter to G and S images, before the dc threshold.
    :param histeq: equalize histogram used in dc image for a better representation.
    Its only applies for dc when plotting it.
    :param nbit: bits oof the image
    :param dc: average intensity image. ndarray
    :param g: image. ndarray. Contains the real coordinate G of the phasor
    :param s: image. ndarray. Contains the imaginary coordinate S of the phasor
    :return: fig: figure contains the avg, histogram, phasor and pseudocolor image.
    """
    if histeq:
        from skimage.exposure import equalize_adapthist
        auxdc = equalize_adapthist(dc / dc.max())
    else:
        auxdc = dc
    if filt:
        from skimage.filters import median
        for i in range(nfilt):
            g = median(g)
            s = median(s)
            phase = median(phase)
            modulation = median(modulation)
    nbit = 2 ** nbit

    # First figure plots dc image and its histogram
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    ax1.imshow(auxdc, cmap='gray')
    ax1.axis('off')
    ax1.set_title('Average intensity image')
    ax2.hist(dc.flatten(), bins=nbit, range=(0, nbit))
    ax2.set_yscale("log")
    ax2.set_title('Average intensity image histogram')
    cursor = Cursor(ax2, horizOn=False, vertOn=True, color='darkgoldenrod')
    ic = plt.ginput(1, timeout=0)
    ic = int(ic[0][0])
    x, y = hsitools.histogram_thresholding(dc, g, s, ic)
    phase = np.where(dc > ic, phase, np.zeros(phase.shape))
    if modulation.any():
        modulation = np.where(dc > ic, modulation, np.zeros(modulation.shape))

    # Phasor plot
    fig2, ax3 = plt.subplots(1, 1, figsize=(8, 6))
    phasor_circle(ax3)
    ax3.set_title('Phasor Plot', pad=20)
    plt.sca(ax3)
    plt.xticks([-1, 0, 1], ['-1', '0', '1'])
    plt.yticks([-1, 0, 1], ['-1', '0', '1'])
    aux = plt.hist2d(x, y, bins=256, cmap="RdYlGn_r", norm=colors.LogNorm(), range=[[-1, 1], [-1, 1]])
    fig2.colorbar(aux[3], ax=ax3)
    ax3.axis('off')

    # Pseudocolor ph-md image and colorbar
    pseudocolor = hsitools.phase_modulation_image(phase, phint, md=modulation, mdinterval=mdint)
    auxphase = np.asarray(np.meshgrid(np.arange(phint[0], phint[1]), np.arange(phint[0], phint[1])))[0]
    auxmd = np.asarray(np.meshgrid(np.linspace(mdint[0], mdint[1], abs(phint[1]-phint[0])),
                                   np.linspace(mdint[0], mdint[1], abs(phint[1]-phint[0]))))[0].transpose()
    pseudo_colorbar = hsitools.phase_modulation_image(auxphase, phint, md=auxmd, mdinterval=mdint)

    fig3, (ax4, ax5) = plt.subplots(1, 2, figsize=(14, 6))
    ax4.imshow(pseudocolor)
    ax4.set_title('Pseudocolor Image')
    ax4.axis('off')
    ax5.imshow(pseudo_colorbar)
    ax5.grid()
    plt.sca(ax5)
    plt.xticks(np.round(np.linspace(0, abs(phint[0]-phint[1]), 10)),
               list(np.round(np.linspace(0, abs(phint[0]-phint[1]), 10))) + phint[0])
    plt.yticks(np.round(np.linspace(0, abs(phint[0]-phint[1]), 10)),
               list(np.round(np.linspace(0, abs(mdint[0]-mdint[1]), 10), 2)) + mdint[0])
    ax5.set_title('HSV Scale for pseudocolor image')
    ax5.set_xlabel('Phase [Degrees]')
    ax5.set_ylabel('Modulation')
    plt.show()
    return fig1, fig2, fig3


def get_ellipsis_range(ellipse):
    # Get the center coordinates, width, height, and angle of the ellipse
    center_x, center_y = ellipse.center
    width = ellipse.width
    height = ellipse.height
    angle = ellipse.angle
    # Calculate points on the ellipse's trace
    num_points = 100  # You can adjust the number of points as needed
    theta = np.linspace(0, 2 * np.pi, num_points)
    x_values = center_x + (width / 2) * np.cos(theta) * np.cos(np.radians(angle)) - (height / 2) * np.sin(theta) * np.sin(np.radians(angle))
    y_values = center_y + (width / 2) * np.cos(theta) * np.sin(np.radians(angle)) + (height / 2) * np.sin(theta) * np.cos(np.radians(angle))
    angles = np.arctan2(y_values, x_values) * 180 / np.pi
    return min(angles),max(angles)


def get_cmap_range(g, s, mask, ellipse):
    g_thresholded = np.zeros_like(g)
    g_thresholded[mask] = g[mask]
    s_thresholded = np.zeros_like(s)
    s_thresholded[mask] = s[mask]
    colormap_angles = np.arctan2(s_thresholded, g_thresholded) * 180 / np.pi
    # colormap_distances = np.sqrt(g**2 + s**2)
    # cmap_range=(0, 180)
    masked_colormap_angles = colormap_angles[mask]
    # cmap_range=(np.min(masked_colormap_angles), np.max(colormap_angles))
    # return colormap_angles, cmap_range
    ellipse_range = get_ellipsis_range(ellipse)

    return colormap_angles, ellipse_range




def get_colormap_image(mask, colormap_angles, cmap_range, cmap):
    # Calculate colormap_vals
    colormap_vals = (colormap_angles - cmap_range[0]) / (cmap_range[1] - cmap_range[0])
    # Map combined values to colors using the colormap
    colormap_colors = cmap(colormap_vals)
    colormap_colors[~mask] = [0, 0, 0, 1]  # [R, G, B, Alpha]
    return colormap_colors



def interactive3(dc, g, s, Ro, nbit, histeq=True, ncomp=5, filt=False, nfilt=0, spectrums=False,
                 hsi_stack=None, lamd=None, flim=False):
    """
        This function plot the avg image, its histogram, the phasors and the rbg pseudocolor image.
    To get the phasor the user must pick an intensity cut umbral in the histogram in order to plot the phasor.
    To get the rgb pseudocolor image you must pick three circle in the phasor plot.
    :param nbit: bits of the image.
    :param dc: average intensity image. ndarray
    :param g: image. ndarray. Contains the real coordinate G of the phasor
    :param s: image. ndarray. Contains the imaginary coordinate S of the phasor
    :param Ro: radius of the circle to select pixels in the phasor

    :param lamd: Lamba array containing the wavelenght. numpy array. Optional
    :param hsi_stack: HSI stack to plot the spectrums of each circle regions.
    :param spectrums: set True to plot the average spectrum of each circle. Optional
    :param nfilt: amount of times to filt G and S images. Optional
    :param filt: Apply median filter to G and S images, before the dc threshold. Optional
    :param ncomp: number of cursors to be used in the phasor, and the pseudocolor image. Default 5.
    :param histeq: equalize histogram used in dc image for a better representation.
            Its only applies for dc when plotting it. Optional
    :param flim: set true if analizing flim images. Plots semicircle instead of full circle in phasor plot.
            Optional

    :return: fig: figure contains the avg, histogram, phasor and pseudocolor image.
    """
    if histeq:
        from skimage.exposure import equalize_adapthist
        auxdc = equalize_adapthist(dc / dc.max())
    else:
        auxdc = dc
    if filt:
        from skimage.filters import median
        for i in range(nfilt):
            g = median(g)
            s = median(s)
    nbit = 2 ** nbit
    
    ic, _, mask = hsitools.otsu_threshold(dc)
    x, y = hsitools.histogram_thresholding(dc, g, s, int(ic))

    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    ax1, ax2, ax3, ax5 = axes.flatten()
    ax1.imshow(auxdc, cmap='gray')
    ax1.axis('off')
    ax1.set_title('Average intensity image')
    ax2.hist(dc.flatten(), bins=nbit, range=(0, nbit))
    ax2.set_yscale("log")
    ax2.set_title('Average intensity image histogram')
    ax2.axvline(x=ic, color='red', linestyle='dashed', linewidth=2, label='Threshold')
    cmap = plt.get_cmap('Spectral')  # You can change the colormap here
    g_thresholded = np.zeros_like(g)
    g_thresholded[mask] = g[mask]
    s_thresholded = np.zeros_like(s)
    s_thresholded[mask] = s[mask]
    figp, (ax3, ax5) = plt.subplots(1, 2, figsize=(16, 6))
    if flim:
        colormap_angles = np.arctan2(s_thresholded, g_thresholded-0.5) * 180 / np.pi
        masked_colormap_angles = colormap_angles[mask]
        cmap_range = (np.min(masked_colormap_angles), np.max(colormap_angles))
        print('inter',cmap_range)
        colormap_vals = (colormap_angles - cmap_range[0]) / (cmap_range[1] - cmap_range[0])
        colormap_colors = cmap(colormap_vals)
        colormap_colors[~mask] = [0, 0, 0, 1]
        phasor_semicircle(ax3, cmap, cmap_range)
        phasorbar = ax3.hist2d(g_thresholded.flatten(), s_thresholded.flatten(), bins=512, cmap="RdYlGn_r", norm=colors.LogNorm(),
                                range=[[-1, 1], [-1, 1]])
        ax3.set_ylim(0, 1)
        ax3.set_xlim(0, 1)
        plt.sca(ax3)
        plt.xticks([0, 1], ['0', '1'])
        plt.yticks([0, 0.5], ['0', '0.5'])

        ax5.imshow(colormap_colors)
        plt.show()
    else:
        phasor_circle_cmap(ax3, cmap,cmap_range)
        phasorbar = ax3.hist2d(x, y, bins=256, cmap="RdYlGn_r", norm=colors.LogNorm(),
                            range=[[-1, 1], [-1, 1]])
        plt.sca(ax3)
        plt.xticks([-1, 0, 1], ['-1', '0', '1'])
        plt.yticks([-1, 0, 1], ['-1', '0', '1'])
        ax3.set_title('Phasor', pad=20)
        fig.colorbar(phasorbar[3], ax=ax3)
        ax5.imshow(colormap_colors)
        plt.show()
    return fig





def calculate_ellipsis(x,y,cmap):
    # Combine x and y into a 2D array
    points = np.array([x, y])

    # Calculate the covariance matrix
    cov_matrix = np.cov(points)

    # Compute the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Extract major and minor axes lengths and the rotation angle
    major_axis_length = 2.0 * np.sqrt(5.991 * eigenvalues[0])
    minor_axis_length = 2.0 * np.sqrt(5.991 * eigenvalues[1])
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    # Find the center of the ellipse
    center_x = np.mean(x)
    center_y = np.mean(y)

    # Create an Ellipse object with the calculated parameters
    ellipse = Ellipse((center_x, center_y), width=major_axis_length, height=minor_axis_length, angle=angle, edgecolor=cmap, fc='None', lw=1.5)
    return ellipse


def multiple_phasorplot(dc_list, g_list, s_list, Ro, nbit, histeq=True, ncomp=5, filt=False, nfilt=0, spectrums=False,
                 hsi_stack=None, lamd=None, flim=False, separate_phasor=False, color_mapping = False):
    """
    This function plots the average image, its histogram, the phasors, and the rgb pseudocolor image for multiple images.
    To get the phasor, the user must pick an intensity cut threshold in the histogram for each image.
    To get the rgb pseudocolor image, the user must pick three circles in the phasor plot for each image.

    :param nbit: bits of the image.
    :param dc_list: List of average intensity images. List of ndarrays.
    :param g_list: List of images containing the real coordinate G of the phasor. List of ndarrays.
    :param s_list: List of images containing the imaginary coordinate S of the phasor. List of ndarrays.
    :param Ro: radius of the circle to select pixels in the phasor

    :param lamd: Lambda array containing the wavelength. numpy array. Optional
    :param hsi_stack: HSI stack to plot the spectrums of each circle regions.
    :param spectrums: set True to plot the average spectrum of each circle. Optional
    :param nfilt: amount of times to filter G and S images. Optional
    :param filt: Apply median filter to G and S images, before the dc threshold. Optional
    :param ncomp: number of cursors to be used in the phasor, and the pseudocolor image. Default 5.
    :param histeq: equalize histogram used in dc image for a better representation.
            It's only applied for dc when plotting it. Optional
    :param flim: set True if analyzing flim images. Plots semicircle instead of full circle in phasor plot. Optional
    :param separate_phasor: set True if you want phasor from different images to be shown in separate colors. Otherwise all phasor are plot together as one
    :return: None
    """
    num_images = len(dc_list)
    
    # Create subplots for average images and histograms
    fig, axes = plt.subplots(2, num_images, figsize=(16, 6))
    ax1 = axes[0]
    ax2 = axes[1]
    nbit = 2 ** nbit
    
    # Create a single phasor plot
    figp, ax3 = plt.subplots(1, 1, figsize=(10, 7))
    fig2, ax4 = plt.subplots(1, num_images, figsize=(16, 6))
    phasor_circle(ax3)
    cmaps = ['Purples','Oranges','Reds','Greens','Blues']
    line_colors = ['magenta','yellow','red','green','blue']
    x_all, y_all = np.empty(0), np.empty(0)
    min_cmap_range = 360
    max_cmap_range = 0
    colormap_angles_all = []
    masks_all = []
    for i in range(num_images):
        dc = dc_list[i]
        g = g_list[i]
        s = s_list[i]
        
        if histeq:
            from skimage.exposure import equalize_adapthist
            auxdc = equalize_adapthist(dc / dc.max())
        else:
            auxdc = dc
        if filt:
            from skimage.filters import median
            for j in range(nfilt):
                g = median(g)
                s = median(s)
        dc = dc.astype(float)
        # Plot average image and histogram side by side
        ax1[i].imshow(auxdc, cmap='gray')
        ax1[i].axis('off')
        ax1[i].set_title('Average intensity image {}'.format(i + 1))
        ax2[i].hist(dc.flatten(), bins=nbit, range=(0, nbit), alpha=0.5)
        ax2[i].set_yscale("log")
        ax2[i].set_title('Average intensity image histogram {}'.format(i + 1))
        ic, _, mask = hsitools.otsu_threshold(dc)
        ax2[i].axvline(x=ic, color='red', linestyle='dashed', linewidth=2, label='Threshold')
        x, y = hsitools.histogram_thresholding(dc, g, s, int(ic))
        if separate_phasor:
            # Overlay phasor plots
            alpha_value = max(0.5, 1 - i / 2) 
            phasorbar = ax3.hist2d(x, y, bins=512, cmap=cmaps[i], norm=colors.LogNorm(),
                                        range=[[-1, 1], [-1, 1]], alpha=alpha_value, label='Image {}'.format(i + 1))
            # phasorbar = ax3.scatter(x, y, c=line_colors[i], s=0.5 ,label='Image {}'.format(i + 1), alpha=alpha_value)
            ellipse = calculate_ellipsis(x,y,line_colors[i])
            ellipse.set_alpha(0.7)
            ax3.add_patch(ellipse)
            colormap_angles , cmap_range = get_cmap_range(g,s,mask, ellipse)
            masks_all.append(mask)
            colormap_angles_all.append(colormap_angles)
            x_all = np.concatenate((x_all, x))
            y_all = np.concatenate((y_all, y))
        else:
            x_all = np.concatenate((x_all, x))
            y_all = np.concatenate((y_all, y))

    ellipse_all = calculate_ellipsis(x_all,y_all,'gray')
    # ax3.add_patch(ellipse_all)
    cmap_range_all = get_ellipsis_range(ellipse_all)
    print('min',cmap_range_all[0])
    print('max',cmap_range_all[1])

            
    if not separate_phasor:
        phasorbar = ax3.hist2d(x_all, y_all, bins=256, cmap="RdYlGn_r", norm=colors.LogNorm(),
                        range=[[-1, 1], [-1, 1]])
    ax3.set_title('Phasor', pad=20)
    plt.sca(ax3)
    plt.xticks([-1, 0, 1], ['-1', '0', '1'])
    plt.yticks([-1, 0, 1], ['-1', '0', '1'])
    # fig.colorbar(phasorbar[3], ax=ax3)
    if color_mapping:
        cmap = plt.get_cmap('Spectral')  # You can change the colormap here
        # cmap_range = (min_cmap_range, max_cmap_range)
        plot_colormap(ax3, cmap_range_all, cmap, zorder=-100)
        for i in range(num_images):
            mask = masks_all[i]
            colormap_angles = colormap_angles_all[i]
            colored_image = get_colormap_image(mask, colormap_angles, cmap_range_all, cmap)
            ax4[i].imshow(colored_image)
    else:
        center = plt.ginput(ncomp, timeout=0)  # get the circle centers
        ccolor = ['red', 'blue', 'green', 'yellow', 'darkviolet']
        for i in range(ncomp):
            circle = plt.Circle((center[i][0], center[i][1]), Ro, color=ccolor[i], fill=False)
            ax3.add_patch(circle)
        for i in range(num_images):
            dc = dc_list[i]
            g = g_list[i]
            s = s_list[i]
            g = np.where(dc > ic, g, dc * np.nan)
            s = np.where(dc > ic, s, dc * np.nan)
            rgba = hsitools.pseudocolor_image(dc, g, s, center, Ro, ncomp=ncomp)
            ax4[i].imshow(rgba)
            ax4[i].set_title('Pseudocolor image of {}'.format(i + 1))
            ax4[i].axis('off')
    plt.show()