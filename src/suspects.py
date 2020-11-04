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


def _download_cluster(msv_id: str, ftp_prefix: str, max_tries: int = 5) \
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
            time.sleep(np.random.uniform(
                high=2 ** (max_tries - tries_left) / 10))
    else:
        logger.warning('Failed to retrieve dataset %s after %d retries',
                       msv_id, max_tries)

    return None, None, None


def _filter_ids(ids: pd.DataFrame, max_ppm: float, min_shared_peaks: int) \
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


def _filter_pairs(pairs: pd.DataFrame, min_cosine: float) -> pd.DataFrame:
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


def _filter_clusters(cluster_info: pd.DataFrame) -> pd.DataFrame:
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


def _generate_suspects(ids: pd.DataFrame, pairs: pd.DataFrame,
                       clusters: pd.DataFrame) -> pd.DataFrame:
    """
    Generate suspects from identifications and aligned spectra pairs.
    Provenance about the spectra pairs is added from the summary.

    Arguments
    ---------
    ids : pd.DataFrame
        The filtered identifications.
    pairs : pd.DataFrame
        The filtered pairs.
    clusters : pd.DataFrame
        The filtered clustering information.

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

    # Add provenance information for the library and suspect scans.
    suspects = (suspects[['Dataset', 'INCHI', 'Compound_Name', 'Adduct',
                          'Cosine', 'Precursor_MZ', 'SpectrumID', '#Scan#',
                          'SuspectIndex']]
                .rename(columns={'Compound_Name': 'CompoundName',
                                 'Precursor_MZ': 'LibraryPrecursorMZ',
                                 'SpectrumID': 'LibraryID',
                                 '#Scan#': 'ClusterScanNr'}))
    suspects = (pd.merge(suspects, clusters,
                         left_on=['Dataset', 'SuspectIndex'],
                         right_on=['Dataset', 'cluster index'])
                .drop(columns=['SuspectIndex', 'cluster index'])
                .rename(columns={'parent mass': 'SuspectPrecursorMZ',
                                 'Original_Path': 'SuspectPath',
                                 'ScanNumber': 'SuspectScanNr'}))
    suspects['DeltaMZ'] = (suspects['SuspectPrecursorMZ'] -
                           suspects['LibraryPrecursorMZ'])
    suspects['GroupDeltaMZ'] = np.nan
    return suspects


def _group_mass_shifts(
        suspects: pd.DataFrame, mass_shift_annotations: pd.DataFrame,
        interval_width: float, bin_width: float, peak_height: float,
        max_dist: float) -> pd.DataFrame:
    """
    Group close mass shifts.

    Mass shifts are binned and the group delta m/z is detected by finding
    peaks in the histogram. Grouped mass shifts are assigned potential
    explanations from the given mass shift annotations. If no annotation can be
    found for a certain group, the rationale and atomic difference will be
    marked as "unspecified". Ungrouped suspects will have a missing rationale
    and atomic difference.

    Arguments
    ---------
    suspects : pd.DataFrame
        The suspects from which mass shifts are grouped.
    mass_shift_annotations : pd.DataFrame
        Mass shift explanations.
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
        The suspects with grouped mass shifts and corresponding rationale (if
        applicable).
    """
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
                 - abs(delta_mz)).abs() < max_dist / 2].sort_values(
                ['priority', 'atomic difference', 'rationale'])
            if len(putative_id) == 0:
                suspects.loc[mask_delta_mz, 'AtomicDifference'] = 'unspecified'
                suspects.loc[mask_delta_mz, 'Rationale'] = 'unspecified'
            else:
                suspects.loc[mask_delta_mz, 'AtomicDifference'] = '|'.join(
                    putative_id['atomic difference'].fillna('unspecified'))
                suspects.loc[mask_delta_mz, 'Rationale'] = '|'.join(
                    putative_id['rationale'].fillna('unspecified'))

    return (suspects.sort_values(['CompoundName', 'Adduct', 'GroupDeltaMZ'])
            .reset_index(drop=True)
            [['Dataset', 'INCHI', 'CompoundName', 'Adduct', 'DeltaMZ',
              'GroupDeltaMZ', 'AtomicDifference', 'Rationale', 'Cosine',
              'LibraryPrecursorMZ', 'LibraryID', 'ClusterScanNr',
              'SuspectPrecursorMZ', 'SuspectScanNr', 'SuspectPath']])


def generate_suspects() -> None:
    """
    Generate suspects from the GNPS living data results.

    Suspect (unfiltered and filtered, unique) metadata is exported to csv files
    in the data directory.

    Settings for the suspect generation are taken from the config file.
    """
    # Expert-based mass shift annotations.
    mass_shift_annotations = pd.read_csv(config.mass_shift_annotation_url)
    mass_shift_annotations['mz delta'] = (mass_shift_annotations['mz delta']
                                          .astype(float))
    mass_shift_annotations['priority'] = (mass_shift_annotations['priority']
                                          .astype(int))

    # Get all GNPS living data cluster results.
    ftp_prefix = f'ftp://massive.ucsd.edu/{config.living_data_base_url}'
    # Get the MassIVE IDs for all datasets included in the living data.
    ftp = ftplib.FTP('massive.ucsd.edu')
    ftp.login()
    ftp.cwd(f'{config.living_data_base_url}/CLUSTERINFO')
    msv_ids = [filename[:filename.find('_')] for filename in ftp.nlst()]
    # Generate the suspects.
    logger.info('Retrieve cluster information')
    ids_pairs_clusters = joblib.Parallel(n_jobs=5)(
        joblib.delayed(_download_cluster)(msv_id, ftp_prefix)
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
    ids = _filter_ids(ids, config.max_ppm, config.min_shared_peaks)
    pairs = _filter_pairs(pairs, config.min_cosine)
    clusters = _filter_clusters(clusters)
    suspects_unfiltered = _generate_suspects(ids, pairs, clusters)
    suspects_unfiltered.to_csv('../../data/suspects_unfiltered.csv',
                               index=False)

    # Ignore suspects without a mass shift.
    suspects_grouped = suspects_unfiltered[
        suspects_unfiltered['DeltaMZ'].abs() > config.min_delta_mz].copy()
    # Group and assign suspects by observed mass shift.
    logger.info('Group suspects by mass shift and assign potential rationales')
    suspects_grouped = _group_mass_shifts(
        suspects_grouped, mass_shift_annotations, config.interval_width,
        config.bin_width, config.peak_height, config.max_dist)
    suspects_grouped.to_csv('../../data/suspects_grouped.csv', index=False)
    # Ignore ungrouped suspects.
    suspects_grouped = suspects_grouped.dropna(subset=['GroupDeltaMZ'])
    # Only use the top suspect (by cosine score) per combination of library
    # spectrum and grouped mass shift.
    suspects_unique = (
        suspects_grouped.sort_values(['Cosine'], ascending=False)
        .drop_duplicates(['CompoundName', 'Adduct', 'GroupDeltaMZ']))
    suspects_unique.to_csv('../../data/suspects_unique.csv', index=False)

    logger.info('%d suspects with non-zero mass differences collected '
                '(%d total)', len(suspects_grouped), len(suspects_unfiltered))
    logger.info('%d unique suspects after duplicate removal and filtering',
                len(suspects_unique))


if __name__ == '__main__':
    logging.basicConfig(
        format='{asctime} [{levelname}/{processName}] {message}',
        style='{', level=logging.INFO)
    logging.captureWarnings(True)
    logger = logging.getLogger('suspect_list')
    logger.setLevel(logging.INFO)

    generate_suspects()
