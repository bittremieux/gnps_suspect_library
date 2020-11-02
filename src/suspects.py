import ftplib
import logging
import math
import time
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import scipy.signal as ssignal
import tqdm

import config


def download_cluster(msv_id: str, ftp_prefix: str, max_tries: int = 5) \
        -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame],
                 Optional[pd.DataFrame]]:
    """
    Download cluster information for the living data analysis with the given
    MassIVE identifier.

    Parameters
    ----------
    msv_id : str
        The MassIVE identifier of the dataset in the living data analysis.
    ftp_prefix : str
        The FTP prefix of the living data results.
    max_tries : int
        The maximum number of times to try downloading files.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple of the identifications, pairs, and clustering DataFrames.
    """
    tries_left = max_tries
    while tries_left > 0:
        try:
            identifications = pd.read_csv(
                f'{ftp_prefix}/IDENTIFICATIONS/'
                f'{msv_id}_identifications.tsv',
                sep='\t', usecols=[
                    'Compound_Name', 'Adduct', 'Precursor_MZ', 'INCHI',
                    'SpectrumID', '#Scan#', 'MZErrorPPM', 'SharedPeaks'])
            identifications['Dataset'] = msv_id
            pairs = pd.read_csv(
                f'{ftp_prefix}/PAIRS/{msv_id}_pairs.tsv', sep='\t',
                usecols=['CLUSTERID1', 'CLUSTERID2', 'Cosine'])
            pairs['Dataset'] = msv_id
            clustering = pd.read_csv(
                f'{ftp_prefix}/CLUSTERINFO/{msv_id}_clustering.tsv',
                sep='\t', usecols=[
                    'cluster index', 'sum(precursor intensity)',
                    'parent mass', 'Original_Path', 'ScanNumber'])
            clustering['Dataset'] = msv_id

            return identifications, pairs, clustering
        except ValueError:
            logger.warning('Error while attempting to retrieve dataset %s',
                           msv_id)
            break
        except IOError:
            tries_left -= 1
            # Exponential back-off.
            time.sleep(2 ** (max_tries - tries_left) / 10)
    else:
        logger.warning('Failed to retrieve dataset %s after %d retries',
                       msv_id, max_tries)

    return None, None, None


def filter_ids(ids: pd.DataFrame, max_ppm: float, min_shared_peaks: int) \
        -> pd.DataFrame:
    """
    Filter high-quality identifications according to the given maximum ppm
    deviation and minimum number of shared peaks.

    Arguments
    ---------
    ids : pd.DataFrame
        The tabular identifications retrieved from GNPS.
    max_ppm : float
        The maximum ppm deviation.
    min_shared_peaks : int
        The minimum number of shared peaks.

    Returns
    -------
    pd.DataFrame
        The identifications retained after filtering.
    """
    return ids[(ids['MZErrorPPM'].abs() <= max_ppm) &
               (ids['SharedPeaks'] >= min_shared_peaks)]


def filter_pairs(pairs: pd.DataFrame, min_cosine: float) -> pd.DataFrame:
    """
    Only consider pairs with a cosine similarity that exceeds the given cosine
    threshold.

    Arguments
    ---------
    pairs : pd.DataFrame
        The tabular pairs retrieved from GNPS.
    min_cosine : float
        The minimum cosine used to retain high-quality pairs.

    Returns
    -------
    pd.DataFrame
        The pairs filtered by minimum cosine similarity.
    """
    return pairs[pairs['Cosine'] >= min_cosine]


def filter_clusters(cluster_info: pd.DataFrame) -> pd.DataFrame:
    """
    For each cluster select as representative the scan with the highest
    precursor intensity.

    Arguments
    ---------
    cluster_info : pd.DataFrame
        The tabular cluster info retrieved from GNPS.

    Returns
    -------
    pd.DataFrame
        Clusters without duplicated spectra by keeping only the scan with the
        highest precursor intensity for each cluster.
    """
    cluster_info = (
        cluster_info.reindex(cluster_info.groupby(
            ['Dataset', 'cluster index'])['sum(precursor intensity)'].idxmax())
        .dropna().reset_index(drop=True)
        [['Dataset', 'cluster index', 'parent mass', 'ScanNumber',
          'Original_Path']])
    cluster_info['cluster index'] = cluster_info['cluster index'].astype(int)
    cluster_info['ScanNumber'] = cluster_info['ScanNumber'].astype(int)
    return cluster_info


def generate_suspects(ids: pd.DataFrame, pairs: pd.DataFrame,
                      summary: pd.DataFrame) -> pd.DataFrame:
    """
    Generate suspects from identifications and aligned spectra pairs.
    Provenance about the spectra pairs is added from the summary.

    Arguments
    ---------
    ids : pd.DataFrame
        The filtered identifications.
    pairs : pd.DataFrame
        The filtered pairs.
    summary : pd.DataFrame
        The filtered summary information for the clusters.

    Returns
    -------
    pd.DataFrame
        A DataFrame with information about both spectra forming a suspect
        identification.
    """
    # Form suspects of library and unidentified spectra pairs.
    suspects = pd.concat([
        pd.merge(pairs, ids,
                 left_on=['Dataset', 'CLUSTERID1'],
                 right_on=['Dataset', '#Scan#'])
        .drop(columns=['CLUSTERID1'])
        .rename(columns={'CLUSTERID2': 'SuspectIndex'}),
        pd.merge(pairs, ids,
                 left_on=['Dataset', 'CLUSTERID2'],
                 right_on=['Dataset', '#Scan#'])
        .drop(columns=['CLUSTERID2'])
        .rename(columns={'CLUSTERID1': 'SuspectIndex'})],
        ignore_index=True, sort=False)

    # TODO: Properly handle this warning.
    if not suspects['SuspectIndex'].is_unique:
        logger.warning('Multiple analog matches per suspect scan found')

    # Add provenance information for the library and suspect scans.
    suspects = (suspects[['Dataset', 'INCHI', 'Compound_Name', 'Adduct',
                          'Cosine', 'Precursor_MZ', 'SpectrumID', '#Scan#',
                          'SuspectIndex']]
                .rename(columns={'Compound_Name': 'CompoundName',
                                 'Precursor_MZ': 'LibraryPrecursorMZ',
                                 'SpectrumID': 'LibraryID',
                                 '#Scan#': 'ClusterScanNr'}))
    suspects = (pd.merge(suspects, summary,
                         left_on=['Dataset', 'SuspectIndex'],
                         right_on=['Dataset', 'cluster index'])
                .drop(columns=['SuspectIndex', 'cluster index'])
                .rename(columns={'parent mass': 'SuspectPrecursorMZ',
                                 'Original_Path': 'SuspectPath',
                                 'ScanNumber': 'SuspectScanNr'}))
    return suspects


def group_mass_shifts(
        suspects: pd.DataFrame, mass_shift_annotations: pd.DataFrame,
        min_delta_mz: float, interval_width: float, bin_width: float,
        peak_height: float, max_dist: float) -> pd.DataFrame:
    """
    Group close mass shifts.

    Mass shifts are binned and the group delta m/z is detected by finding
    peaks in the histogram.

    Arguments
    ---------
    suspects : pd.DataFrame
        The suspects from which mass shifts are grouped.
    mass_shift_annotations : pd.DataFrame
        Mass shift explanations.
    min_delta_mz : float
        The minimum (absolute) delta m/z for suspects to be retained.
    interval_width : float
        The size of the interval in which mass shifts are binned, centered
        around unit masses.
    bin_width : float
        The bin width used to construct the histogram.
    peak_height : float
        The minimum height for a peak to be considered as a group.
    max_dist : float
        The maximum m/z difference that group members can have with the
        group's peak.

    Returns
    -------
    pd.DataFrame
        The suspects with grouped mass shifts.
    """
    suspects['DeltaMZ'] = (suspects['SuspectPrecursorMZ'] -
                           suspects['LibraryPrecursorMZ'])
    # Remove suspects with an insufficient mass shift.
    suspects = suspects[suspects['DeltaMZ'].abs() > min_delta_mz].copy()
    # Assign putative identifications to the mass shifts.
    for mz in np.arange(math.floor(suspects['DeltaMZ'].min()),
                        math.ceil(suspects['DeltaMZ'].max() + interval_width),
                        interval_width):
        suspects_interval = suspects[suspects['DeltaMZ'].between(
            mz - interval_width / 2, mz + interval_width / 2)]
        if len(suspects_interval) == 0:
            continue
        # Get peaks for frequent deltas in the histogram.
        bins = (np.linspace(mz - interval_width / 2,
                            mz + interval_width / 2,
                            int(interval_width / bin_width) + 1)
                + bin_width / 2)
        hist, _ = np.histogram(suspects_interval['DeltaMZ'], bins=bins)
        peaks_i, prominences = ssignal.find_peaks(
            hist, height=peak_height, distance=max_dist / bin_width,
            prominence=(None, None))
        if len(peaks_i) == 0:
            continue
        # Assign deltas to their closest peak.
        mask_peaks = np.unique(np.hstack(
            [suspects_interval.index[suspects_interval['DeltaMZ']
                                     .between(min_mz, max_mz)]
             for min_mz, max_mz in zip(bins[prominences['left_bases']],
                                       bins[prominences['right_bases']])]))
        mz_diffs = np.vstack([
            np.abs(suspects.loc[mask_peaks, 'DeltaMZ'] - peak)
            for peak in bins[peaks_i]])
        # Also make sure that delta assignments don't exceed the maximum
        # distance.
        mask_mz_diffs = mz_diffs.min(axis=0) < max_dist
        mz_diffs = mz_diffs[:, mask_mz_diffs]
        mask_peaks = mask_peaks[mask_mz_diffs]
        peak_assignments = mz_diffs.argmin(axis=0)
        # Assign putative explanations to the grouped mass shifts.
        for delta_mz, peak_i in zip(bins[peaks_i], range(len(peaks_i))):
            mask_delta_mz = mask_peaks[peak_assignments == peak_i]
            suspects.loc[mask_delta_mz, 'GroupDeltaMZ'] = delta_mz
            putative_id = mass_shift_annotations[
                (mass_shift_annotations['mz delta'].abs()
                 - abs(delta_mz)).abs() < max_dist / 2]
            putative_id = putative_id.sort_values(
                ['priority', 'atomic difference', 'rationale'])
            if len(putative_id) == 0:
                suspects.loc[mask_delta_mz, 'AtomicDifference'] = 'unknown'
                suspects.loc[mask_delta_mz, 'Rationale'] = 'unspecified'
            else:
                suspects.loc[mask_delta_mz, 'AtomicDifference'] = '|'.join(
                    putative_id['atomic difference'].fillna('unspecified'))
                suspects.loc[mask_delta_mz, 'Rationale'] = '|'.join(
                    putative_id['rationale'].fillna('unspecified'))
    # Set delta m/z's for ungrouped suspects.
    suspects['GroupDeltaMZ'].fillna(suspects['DeltaMZ'], inplace=True)

    return (suspects.sort_values(['CompoundName', 'GroupDeltaMZ'])
            .reset_index(drop=True)
            [['Dataset', 'INCHI', 'CompoundName', 'Adduct', 'DeltaMZ',
              'GroupDeltaMZ', 'AtomicDifference', 'Rationale', 'Cosine',
              'LibraryPrecursorMZ', 'LibraryID', 'ClusterScanNr',
              'SuspectPrecursorMZ', 'SuspectScanNr', 'SuspectPath']])


if __name__ == '__main__':
    logging.basicConfig(
        format='{asctime} [{levelname}/{processName}] {message}',
        style='{', level=logging.INFO)
    logging.captureWarnings(True)
    logger = logging.getLogger('suspect_list')
    logger.setLevel(logging.INFO)

    # Expert-based mass shift annotations.
    mass_shift_annotations = pd.read_csv(
        'https://docs.google.com/spreadsheets/d/'
        '1-xh2XpSqdsa4yU-ATpDRxmpZEH6ht982jCCATFOpkyM/'
        'export?format=csv&gid=566878567')
    mass_shift_annotations['mz delta'] = (mass_shift_annotations['mz delta']
                                          .astype(np.float64))
    mass_shift_annotations['priority'] = (mass_shift_annotations['priority']
                                          .astype(np.uint8))

    # Get all GNPS living data cluster results.
    ftp_prefix = f'ftp://massive.ucsd.edu/{config.base_url}'
    # Get the MassIVE IDs for all datasets included in the living data.
    ftp = ftplib.FTP('massive.ucsd.edu')
    ftp.login()
    ftp.cwd(f'{config.base_url}/CLUSTERINFO')
    msv_ids = [filename[:filename.find('_')] for filename in ftp.nlst()]
    # Generate the suspects.
    logger.info('Retrieve cluster information')
    ids_pairs_clusters = joblib.Parallel(n_jobs=5)(
        joblib.delayed(download_cluster)(msv_id, ftp_prefix)
        for msv_id in tqdm.tqdm(msv_ids, desc='Datasets processed',
                                unit='dataset'))
    ids, pairs, clusters = [], [], []
    for i, p, c in ids_pairs_clusters:
        if i is not None and p is not None and c is not None:
            ids.append(i)
            pairs.append(p)
            clusters.append(c)
    ids = pd.concat(ids, ignore_index=True)
    pairs = pd.concat(pairs, ignore_index=True)
    clusters = pd.concat(clusters, ignore_index=True)
    # Compile suspects from the clustering data.
    logger.info('Compile suspect pairs')
    ids = filter_ids(ids, config.max_ppm, config.min_shared_peaks)
    pairs = filter_pairs(pairs, config.min_cosine)
    clusters = filter_clusters(clusters)
    suspects_unfiltered = generate_suspects(ids, pairs, clusters)
    suspects_unfiltered.to_csv('../../data/suspects_unfiltered.csv',
                               index=False)

    # Group and assign suspects by observed delta m/z.
    suspects_unfiltered = pd.read_csv('../../data/suspects_unfiltered.csv')
    logger.info('Assign putative explanations to mass shifts')
    suspects = group_mass_shifts(suspects_unfiltered, mass_shift_annotations,
                                 config.min_delta_mz, config.interval_width,
                                 config.bin_width, config.peak_height,
                                 config.max_dist)
    # Only use the top suspect (by cosine score) per combination of library
    # spectrum and putative identification.
    suspects_unique = (suspects.sort_values(['Cosine'], ascending=False)
                       .drop_duplicates(['CompoundName', 'Adduct',
                                         'GroupDeltaMZ']))

    delta_mzs = (suspects['GroupDeltaMZ'].value_counts().reset_index()
                 .rename(columns={'GroupDeltaMZ': 'Count',
                                  'index': 'GroupDeltaMZ'})
                 .sort_values('Count', ascending=False))

    suspects_unique_filtered = (suspects_unique[
        suspects_unique['GroupDeltaMZ'].isin(
            delta_mzs.loc[delta_mzs['Count'] >= config.min_group_size,
                          'GroupDeltaMZ'])])
    suspects_unique_filtered.to_csv('../../data/suspects_unique.csv',
                                    index=False)

    logger.info('Total: %d suspects collected', len(suspects))
    logger.info('After duplicate removal and filtering (delta m/z occurs at '
                'least %d times): %d unique suspects', config.min_group_size,
                len(suspects_unique_filtered))
