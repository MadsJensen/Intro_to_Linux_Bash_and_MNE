
# coding: utf-8

#### import settings


import numpy as np
import mne
from mne.preprocessing import ICA
from mne.minimum_norm import (make_inverse_operator,
                              write_inverse_operator)
import os
# directories ISIS
subjects_dir = "/volatile/mje/training_data/fs_subjects_dir/"
data_path = "/volatile/mje/training_data/"

os.chdir(data_path)

n_jobs = 4 # number of processors to use

# epoch variables
tmin, tmax = -0.5, 0.5
baseline = (-0.5, -0.3)  # baseline time
reject = dict(mag=4e-12, grad=4000e-13)

fs_sub = "0003_WVZ"

# only for ipython notebook
get_ipython().magic(u'matplotlib inline')


### Maxfilter from python

# file and log names

in_name = "sub_%d_%s-raw.fif" % (sub, session)
out_name = "sub_%d_%s-tsss-mc-autobad_ver_4.fif" % (sub, session)
tsss_mc_log = "sub_%d_%s-tsss-mc-autobad_ver_4.log" % (sub, session)
headpos_log = "sub_%d_%s-headpos_ver_4.log" % (sub, session)

# call to maxfilter

apply_maxfilter(in_fname=in_name,
       out_fname=out_name,
       frame='head',
       autobad="on",
       st=True,
       st_buflen=30,
       st_corr=0.95,
       mv_comp=True,
       mv_hp=headpos_log,
       cal='/projects/MINDLAB2011_24-MEG-readiness/misc/sss_cal_Mar11-May13.dat',
       ctc='/projects/MINDLAB2011_24-MEG-readiness/misc/ct_sparse_Mar11-May13.fif',
       overwrite=True,
       mx_args=' -v | tee %s' % tsss_mc_log,
       )


#### Processing raw files


raw = mne.fiff.Raw(data_path +
                          "sef_bilat-tsss-mc-autobad.fif", preload=True)

picks = mne.fiff.pick_types(raw.info, meg=True, eeg=False, eog=True,
                                   emg=True, ecg=True, exclude='bads')


raw.info

##### Filter data
raw.filter(None, 40, method='iir', n_jobs=n_jobs)

events = mne.find_events(raw, stim_channel='STI101')
event_ids = {"Left": 1, "Right": 2}  # check this is right

epochs = mne.Epochs(raw, events, event_ids, tmin, tmax,
                           picks=picks, baseline=(None, -0.2),
                           preload=True, reject=reject)


# In[6]:

mne.viz.plot_drop_log(epochs.drop_log)


#### ICA

ica = ICA(n_components=0.90, n_pca_components=64,
          max_pca_components=100,
          noise_cov=None)

ica.decompose_epochs(epochs)

eog_scores_1_normal = ica.find_sources_epochs(epochs, target="EOG001",
                                              score_func="pearsonr")
eog_scores_2_normal = ica.find_sources_epochs(epochs, target="EOG003",
                                              score_func="pearsonr")

# get maximum correlation index for EOG
eog_source_idx_1_normal = np.abs(eog_scores_1_normal).argmax()
eog_source_idx_2_normal = np.abs(eog_scores_2_normal).argmax()

source_idx = range(0, ica.n_components_)
ica.plot_topomap(source_idx, ch_type="mag")


# select ICA sources and reconstruct MEG signals, compute clean ERFs
# Add detected artefact sources to exclusion list
# We now add the eog artefacts to the ica.exclusion list
if eog_source_idx_1_normal == eog_source_idx_2_normal:
    ica.exclude += [eog_source_idx_1_normal]
elif eog_source_idx_1_normal != eog_source_idx_2_normal:
    ica.exclude += [eog_source_idx_1_normal, eog_source_idx_2_normal]

# remove ECG
ecg_ch_name = 'ECG002'
ecg_scores = ica.find_sources_epochs(epochs, target=ecg_ch_name,
                                     score_func='pearsonr')

# get the source most correlated with the ECG.
ecg_source_idx = np.argsort(np.abs(ecg_scores))[-1]
ica.exclude += [ecg_source_idx]



# Restore sensor space data
# plot spatial sensitivities of a few ICA components
# title = 'Spatial patterns of ICA components (Magnetometers)'
# source_idx = range(0, ica.n_components_)
# ica.plot_topomap(source_idx, ch_type='mag')
# plt.suptitle(title, fontsize=12)

epochs_ica = ica.pick_sources_epochs(epochs)
# epochs_ica.save("sef_bilat-tsss-mc-autobad-epochs.fif")

print epochs_ica
epochs_ica.info


#### Evoked data

evoked_left = epochs_ica["Left"].average()
evoked_right = epochs_ica["Right"].average()

print "plot evoked for Left"
evoked_left.plot()

print "plot evoked for Right"
evoked_right.plot()

# plot all channels
mne.viz.plot_topo([evoked_left, evoked_right],
                  colors=["green", "red"], title="Left vs. Right")

### Estimate noise covariance

cov = mne.compute_covariance(epochs, tmin=None, tmax=-0.2)
mne.viz.plot_cov(cov, raw.info, colorbar=True, proj=True)

# save noise cov
# cov.save("sef_bilat-tsss-mc-autobad.fif")


#### Make forward model (in bash)

# Run the mne_do_forward_model from a termninal

#### load forward model

mri = data_path + "sef_bilat-tsss-mc-autobad-epochs.fif"
fwd = mne.read_forward_solution("sef_bilat-tsss-mc-autobad-fwd.fif")

# convert to surface orientation for better visualization
fwd = mne.convert_forward_solution(fwd, surf_ori=True)


#### Compute inverse solution

# inversion variables
snr = 1.0
lambda2 = 1.0 / snr ** 2
methods = ["dSPM", "MNE"]

for method in methods:
    """ loop over methods to make inverse operator for each method
    """
    inv_op = make_inverse_operator(raw.info, forward=fwd,
                                          noise_cov=cov,
                                          loose=0.2, depth=0.8)
#    write_inverse_operator(
#        "sef_bilat-tsss-mc-autobad%s-inv.fif"
#        % (method), inv_op_normal)

### Bits


events = mne.find_events(raw, stim_channel='STI101')
event_ids = {"press": 1}
events_classic = []
events_interupt = []

for i in range(len(events)):
    if i > 0:
        if events[i, 2] == 1 and events[i - 1, 2] == 1:
            events_classic.append(i)
        elif events[i, 2] == 1 and events[i - 1, 2] == 2:
            events_interupt.append(i)

