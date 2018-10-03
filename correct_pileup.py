# creates the pileup corrected histogram
# also copies the lost muon histogram
#
# For now has hardcoded histogram names and rebin factors
#
# Takes a couple minutes
#
# Aaron Fienberg
# September 2018

import sys
import ROOT as r
import numpy as np
from precessionlib.calospectra import CaloSpectra
from precessionlib.util import rebinned_last_axis

pu_energy_min = 3500

do_errors = True


def build_root_hists(filename, histname, rebin_factor, do_errors):
    '''
    use gm2pileup module to build pileup corrected and non corrected 3d hists
    returns uncorrected, corrected, pileup_normalizations
    pileup normalizations can be used to adjust fit weights in corrected hist
    '''
    spec = CaloSpectra.from_root_file(filename, histname, pu_energy_min=3500)

    spec.set_hist_errors

    rebinned_axes = list(spec.axes)
    rebinned_axes[-1] = rebinned_axes[-1][::rebin_factor]

    rebinned_spec = rebinned_last_axis(spec.array, rebin_factor)

    uncorrected_hist = spec.build_root_hist(
        rebinned_spec, rebinned_axes, 'uncorrected')

    rebinned_corrected = rebinned_spec - \
        rebinned_last_axis(spec.pu_spectrum, rebin_factor)

    corrected_hist = spec.build_root_hist(
        rebinned_corrected, rebinned_axes, 'corrected')

    if do_errors:
        errors = np.sqrt(rebinned_last_axis(
            spec.cor_variances, rebin_factor))
        spec.set_hist_errors(errors, corrected_hist)

    return uncorrected_hist, corrected_hist


def main():
    if len(sys.argv) < 2:
        print('Usage: correct_pileup.py <input root file>')
        return 0

    infile_name = sys.argv[1]
    uncorrected, corrected = build_root_hists(
        infile_name, 'clustersAndCoincidences/clusters', 6, do_errors)

    file = r.TFile(infile_name)
    lost_muon_hist = file.Get('clustersAndCoincidences/triples')
    ctag_hist = file.Get('clustersAndCoincidences/ctag')

    if do_errors:
        outfile_name = infile_name.rstrip(
            '.root') + '_pileup_corrected_errors.root'
    else:
        outfile_name = infile_name.rstrip('.root') + '_pileup_corrected.root'

    outf = r.TFile(outfile_name, 'recreate')

    for hist in [uncorrected, corrected, lost_muon_hist, ctag_hist]:
        hist.SetDirectory(r.gDirectory)
        hist.Write()

    outf.Write()


if __name__ == '__main__':
    sys.exit(main())
