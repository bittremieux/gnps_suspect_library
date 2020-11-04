# Criteria to form a suspect:
#
# - Identification ≤ 20 ppm.
# - Identification ≥ 6 shared peaks.
# - Cosine ≥ 0.8.
# - The spectrum with maximal precursor intensity is chosen as cluster
#   representative.
#
# Criteria to assign group delta m/z's:
#
# - Only delta m/z's that exceed 0.5 Da are considered.
# - Delta m/z's are examined within each 1 m/z window separately
#   (centered around unit m/z's).
# - Delta m/z's are binned with 0.002 bin width.
# - Peaks with minimum height 10 are extracted from the delta m/z histograms.
# - Delta m/z's between the left and right bases of each peak and at maximum
#   0.01 Da distance from the peak m/z's are grouped.
#
# Suspect filtering:
#
# - Suspects whose delta m/z occurs less than 10 times are discarded.

mass_shift_annotation_url = ('https://docs.google.com/spreadsheets/d/'
                             '1-xh2XpSqdsa4yU-ATpDRxmpZEH6ht982jCCATFOpkyM/'
                             'export?format=csv&gid=566878567')

living_data_base_url = 'MSV000084314/updates/2020-10-08_mwang87_d7c866dd/other'

max_ppm = 20
min_shared_peaks = 6
min_cosine = 0.8
min_delta_mz = 0.5
interval_width = 1.0
bin_width = 0.002
peak_height = 10
max_dist = 0.01
