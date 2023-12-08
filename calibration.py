#%%
import tifffile
import numpy as np
import matplotlib
# matplotlib.use('TKAgg')
import hsitools 
import hsi_visualization 
import os
# import matplotlib.pyplot as plt
import lfdfiles as lfd
import json
from src.phasorpy import io
from sklearn.cluster import KMeans
#%%
def centroid(g,s):
    x = g.flatten()
    y = s.flatten()
    X1 = np.zeros([2, len(x)])
    X1[0:, 0:] = x, y
    X = X1.T
    cluster = KMeans(n_clusters=1).fit(X)
    x_center, y_center = cluster.cluster_centers_[0][0], cluster.cluster_centers_[0][1]
    return x_center, y_center

def calibration_factor(calibration_image, reference, freq):
    # Calculate g and s
    _, g, s, _, _ = np.asarray(hsitools.phasor(calibration_image))
    # Get reference g and s 
    with open('references.json', 'r') as json_file:
        references = json.load(json_file)
    reference_tau = float(references[reference])
    omega = 2 * np.pi * freq/1000
    M = 1 / np.sqrt(1 + (omega * reference_tau) **2)
    phi = np.arctan(omega * reference_tau)
    reference_g = M * np.cos(phi)
    reference_s = M * np.sin(phi)
    # reference_g = 1 / (1 + np.power(omega * reference_tau, 2))
    # reference_s = omega * reference_tau / (1 + np.power(omega * reference_tau, 2))
    # Get correction factor according to reference and calculated centroid from data
    centroids = centroid(g,s)
    g_correction_factor = reference_g - centroids[0] # Reference - centroid
    s_correction_factor = reference_s - centroids[1] # Reference - centroid
    return g_correction_factor, s_correction_factor
    # phi_given = np.arctan(centroids[1] / centroids[0])
    # m_given = np.sqrt(centroids[0] ** 2 + centroids[1] ** 2)
    # if centroids[0] < 0:
    #     theta_correction = phi - np.pi - phi_given
    # else:
    #     theta_correction = phi - phi_given
    # m_correction = M / m_given
    # return theta_correction, m_correction

def apply_calibration(image, calibration_image, reference, freq):
    g_correction_factor, s_correction_factor = calibration_factor(calibration_image, reference, freq)
    # theta_correction, m_correction = calibration_factor(calibration_image, reference, freq)
    # Calculate g and s
    dc, g, s, md, ph = np.asarray(hsitools.phasor(image))
    g_calibrated = g + g_correction_factor
    s_calibrated = s + s_correction_factor
    # md = np.sqrt(g**2+s**2)*m_correction
    # ph = np.arctan(s/g)+theta_correction
    # md = md*m_correction
    # ph = ph + theta_correction 
    # g_calibrated = md*np.cos(ph)
    # s_calibrated = md*np.sin(ph)
    return dc, g_calibrated, s_calibrated

#%%
#Open calibration files and calculate centroid of data
# calibration_file_path = 'test_data/flim/calibration/convularia/Ex780p64_BH_g2_coumarin_c100_1000.Z64'
# image_file_path = 'test_data/flim/calibration/convularia/Ex780p64_BH_g2_convularia_c400_1000.Z64'
calibration_file_path = '/Volumes/BRUNODATA/FLIMBox vs BH project/Suman Leonel 01302018 BH/Ex780p60_BH_coumarine_c100_1000.Z64'
image_file_path = '/Volumes/BRUNODATA/FLIMBox vs BH project/Suman Leonel 01302018 BH/Ex780p95_BH_Rh110_c500_1000.Z64'
# image_file_path = '/Volumes/BRUNODATA/FLIMBox vs BH project/Suman Leonel 01302018 BH/Ex780p62_BH__9_10H_acridinone_c50_1000.Z64'
# calibration_file_path = 'test_data/flim/Suman Leonel 01302018 BH/Ex780p60_BH_coumarine_c100_1000.Z64'
# image_file_path = 'test_data/flim/Suman Leonel 01302018 BH/Ex780p60_BH_9_10H_acridinone_Rh110_cy3_c100_1000.Z64'
calibration_image = io.read_z64(calibration_file_path)
image = io.read_z64(image_file_path)
dc, g_calibrated, s_calibrated = apply_calibration(image = image, calibration_image = calibration_image, reference = 'coumarin_6', freq = 80)
# dc, g_calibrated, s_calibrated = apply_calibration(image = image, calibration_image = calibration_image, reference = 'coumarin_6', freq = 62.5)
hsi_visualization.interactive3(dc, g_calibrated, s_calibrated, 0.15, 8, ncomp=3, nfilt=3,filt=True, spectrums=False,
                                hsi_stack=calibration_image, lamd=np.linspace(418, 718, 30),flim=True)
# hsi_visualization.interactive1(dc, g_calibrated, s_calibrated, 0.15, 8, ncomp=3, nfilt=3, spectrums=False,
#                                 hsi_stack=image, lamd=np.linspace(418, 718, 30))

#%%
# calibration = '/Volumes/JD Drive/Data_dem_workshop/Data_11-23/coumarin6_000$EI0S.fbd'
# calibration = '/Volumes/JD Drive/Data_dem_workshop/Data_11-23/RH110CALIBRATION_000$EI0S.fbd'
# convallaria = '/Volumes/JD Drive/Data_dem_workshop/Data_11-23/convallaria_000$EI0S.fbd'
calibration = '/Volumes/JD Drive/Data_dem_workshop/Nov15_23/10x800n764uP13RHO110_111$EI0S.fbd'
convallaria = '/Volumes/JD Drive/Data_dem_workshop/Nov15_23/10x800n764uP6convallaria_000$EI0S.fbd'
with lfd.SimfcsFbd(calibration) as cal, lfd.SimfcsFbd(convallaria) as conv:
    bins_times_markers = cal.decode()
    frames_cal = cal.frames(bins_times_markers)
    cal_image = cal.asimage(bins_times_markers, frames_cal)
    cal_image = cal_image[0,0,:,:,:]
    cal_image = np.transpose(cal_image, (2, 0, 1))
    bins_times_markers = conv.decode()
    frames_conv = conv.frames(bins_times_markers)
    conv_image = conv.asimage(bins_times_markers, frames_conv)
    conv_image = conv_image[0,0,:,:,:]
    conv_image = np.transpose(conv_image, (2, 0, 1))

    dc, g_calibrated, s_calibrated = apply_calibration(image = conv_image, calibration_image = cal_image, reference = 'rhodamine_110', freq = 80)
    # dc, g_calibrated, s_calibrated, _ , _ = hsitools.phasor(image)
    hsi_visualization.interactive3(dc, g_calibrated, s_calibrated, 0.15, 8, ncomp=3, nfilt=3,filt=True, spectrums=False,
                                hsi_stack=conv_image, lamd=np.linspace(418, 718, 30),flim=True)
# %%