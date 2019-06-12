# functions for running omega_a systematic sweeps
#
# Aaron Fienberg
# June 2019

import json
import ROOT as r
import numpy as np
from .calospectra import CaloSpectra
from .util import *
from .fitmodels import *
from .analysis import *

#
# Pileup phase sweep code
#


def pileup_phase_sweep(config, conf_dir):
    ''' run the pileup phase sensitivity scan '''
    print('\n---\nRunning pileup phase sweep\n---')

    raw_f, fit_info = load_raw_and_analyzed_files(config, conf_dir)

    print('Building calo spectrum and pileup correction...')
    hist_name = config['hist_name']
    dir_name = hist_name.split('/')[0]
    ctag = raw_f.Get(f'{dir_name}/ctag')
    n_fills = ctag.GetEntries()
    print(f'{n_fills:.0f} fills')

    spec = CaloSpectra.from_root_file(f'{conf_dir}/{config["raw_file"]}',
                                      config['hist_name'],
                                      # do_triple=False,
                                      do_triple=True,
                                      single_param=True,
                                      param_guess=2 * 1.25 / n_fills / 25)
    print(f'estimated deadtimes per calo: {spec.estimated_deadtimes(n_fills)}')

    # run the scan
    print('running pileup time shift scan...')
    outs_T = []
    outs_A = []

    shifts = list(range(config['shift_min'], config['shift_max'] + 1))
    for shift in shifts:
        print(f'shift {shift}....')
        out_T, out_A = fit_pu_shifted_T_and_A(shift, spec, fit_info, config)
        outs_T.append(out_T)
        outs_A.append(out_A)

    # convert shifts to units of microseconds
    for i, shift in enumerate(shifts):
        shifts[i] = shifts[i] * (spec.time_centers[1] - spec.time_centers[0])

    # print out the sensitivity
    time_range = (shifts[-1] - shifts[0])

    T_R_range = outs_T[-1][1].GetParameter(4) - outs_T[0][1].GetParameter(4)
    T_sens = T_R_range / time_range

    A_R_range = outs_A[-1][1].GetParameter(4) - outs_A[0][1].GetParameter(4)
    A_sens = A_R_range / time_range

    print('\nPhase scan summary:')
    print(f'T-Method sensitivity: {T_sens:.3f} ppb / ns')
    print(f'A-Weighted sensitivity: {A_sens:.3f} ppb / ns')

    return outs_T, outs_A, shifts


def fit_pu_shifted_T_and_A(shift_in_bins, spec, fit_info, config, rebin=6):
    _ = r.TCanvas()

    hist_T, hist_A = build_pu_shifted_T_and_A(
        shift_in_bins, spec, fit_info, rebin)

    return fit_T_and_A(hist_T, hist_A, fit_info,
                       config, f'pu_shift_{shift_in_bins}')


def build_pu_shifted_T_and_A(shift_in_bins, spec, fit_info, rebin=6):
    ''' build T-Method and A-Weighted histograms with the pileup spectrum
    shifted by a certain number of bins'''

    uncor_spec = spec.array
    pu_spec = spec.pu_spectrum

    shifted_pu = np.zeros_like(pu_spec)

    if shift_in_bins > 0:
        shifted_pu[:, :, shift_in_bins:] = pu_spec[:, :, :-shift_in_bins]
    elif shift_in_bins < 0:
        shifted_pu[:, :, :shift_in_bins] = pu_spec[:, :, -shift_in_bins:]
    else:
        shifted_pu[:, :, :] = pu_spec[:, :, :]

    shifted_corrected = rebinned_last_axis(uncor_spec - shifted_pu,
                                           rebin)
    rebinned_axes = list(spec.axes)
    rebinned_axes[-1] = spec.axes[-1][::rebin]

    shifted_3d = CaloSpectra.build_root_hist(shifted_corrected,
                                             rebinned_axes,
                                             f'pushifted3d_{shift_in_bins}')

    master_2d = shifted_3d.Project3D(f'yx_ps{shift_in_bins}')

    T_hist = master_2d.ProjectionX(f'TPUShifted{shift_in_bins}',
                                   fit_info['thresh_bin'], -1)

    A_hist = build_a_weight_hist(master_2d, fit_info['a_model'],
                                 f'APUShifted{shift_in_bins}')

    apply_pu_unc_factors(T_hist, A_hist, fit_info)

    return T_hist, A_hist

#
# Approximate IFG sweep code
#


def ifg_amplitude_sweep(config, conf_dir):
    ''' run the IFG amplitude sweep '''
    print('\n---\nRunning approximate IFG amplitude sweep\n---')

    raw_f, fit_info = load_raw_and_analyzed_files(config, conf_dir)

    print('Building corrected, uncorrected, and difference hists...')
    cor = raw_f.Get(config['cor_hist_name'])
    cor_2d = cor.Project3D('yx_1')
    cor_2d.SetName('IFGCorrected2D')

    uncor = raw_f.Get(config['uncor_hist_name'])
    uncor_2d = uncor.Project3D('yx_2')
    uncor_2d.SetName('IFGUncorrected2D')

    delta_rho = uncor_2d.Clone()
    delta_rho.SetName('delta_rho')
    delta_rho.Add(cor_2d, -1)

    # run the scan
    print('running approximate ifg amplitude scan...')
    if config['scale_max'] < config['scale_min']:
        raise RuntimeError(
            'Problem with IFG scan configuration!'
            '"scale_min" must be less than "scale_max"')

    scales = []
    scale = config['scale_min']
    while scale <= config['scale_max']:
        scales.append(scale)
        scale += config['scale_step']

    outs_T = []
    outs_A = []
    for scale in scales:
        print(f'scale {scale}....')
        out_T, out_A = fit_ifg_scaled_T_and_A(
            scale, cor_2d, delta_rho, fit_info, config)
        outs_T.append(out_T)
        outs_A.append(out_A)

    scale_range = scales[-1] - scales[0]
    T_R_range = outs_T[-1][1].GetParameter(4) - outs_T[0][1].GetParameter(4)
    T_sens = T_R_range / scale_range

    A_R_range = outs_A[-1][1].GetParameter(4) - outs_A[0][1].GetParameter(4)
    A_sens = A_R_range / scale_range

    print('\nIFG amplitude scan summary:')
    print(f'T-Method sensitivity: {T_sens*1000:.1f} ppb / IFG correction')
    print(f'A-Weighted sensitivity: {A_sens*1000:.1f} ppb / IFG correction')

    return outs_T, outs_A, scales


def build_ifg_scaled_T_and_A(scale_factor, cor_2d, delta_rho, fit_info):
    scaled_2d = cor_2d.Clone()
    scaled_2d.SetName(f'scaled2D_{scale_factor}')
    scaled_2d.Add(delta_rho, float(1 - scale_factor))

    T_hist = scaled_2d.ProjectionX(f'TScaled{scale_factor}',
                                   fit_info['thresh_bin'], -1)

    A_hist = build_a_weight_hist(
        scaled_2d, fit_info['a_model'], f'AScaled{scale_factor}')

    apply_pu_unc_factors(T_hist, A_hist, fit_info)

    return T_hist, A_hist


def fit_ifg_scaled_T_and_A(scale_factor, cor_2d, delta_rho, fit_info, config):
    _ = r.TCanvas()

    hist_T, hist_A = build_ifg_scaled_T_and_A(
        scale_factor, cor_2d, delta_rho, fit_info)

    return fit_T_and_A(hist_T, hist_A, fit_info,
                       config, f'ifg_scale_{scale_factor}')

#
# Common code
#


def fit_T_and_A(hist_T, hist_A, fit_info, config, name):
    fit_T = configure_model_fit(
        fit_info['example_T_fit'], f'TFit_{name}', config)

    resids_T, fft_T = fit_and_fft(hist_T, fit_T, f'fullFitScale{name}',
                                  config['fit_options'],
                                  fit_info['fit_start'],
                                  fit_info['fit_end'],
                                  double_fit=True)

    fit_A = configure_model_fit(
        fit_info['example_A_fit'], f'AFit_{name}', config)

    resids_A, fft_A = fit_and_fft(hist_A, fit_A,
                                  f'AFitScale{name}',
                                  config['fit_options'],
                                  fit_info['fit_start'],
                                  fit_info['fit_end'],
                                  double_fit=True)

    return (hist_T, fit_T, fft_T), (hist_A, fit_A, fft_A)


def apply_pu_unc_factors(T_hist, A_hist, fit_info):
    ''' apply pileup uncertainty factors to the T and A hists '''
    T_meth_unc_facs = fit_info['T_meth_unc_facs']
    A_weight_unc_facs = fit_info['A_weight_unc_facs']

    # pu uncertainty factors
    if len(T_meth_unc_facs):
        for i_bin in range(1, T_hist.GetNbinsX() + 1):
            #  don't take sqrt of negative bin contents
            if T_hist.GetBinContent(i_bin) > 0:
                old_err = np.sqrt(T_hist.GetBinContent(i_bin))
                new_err = old_err * T_meth_unc_facs[i_bin - 1]
                T_hist.SetBinError(i_bin, new_err)

    if len(A_weight_unc_facs):
        for i_bin in range(1, A_hist.GetNbinsX() + 1):
            old_err = A_hist.GetBinError(i_bin)
            new_err = old_err * A_weight_unc_facs[i_bin - 1]
            A_hist.SetBinError(i_bin, new_err)


def load_pu_uncertainties(pu_unc_file):
    ''' load pileup uncertainties from the text file '''
    if pu_unc_file is not None:
        factor_array = np.loadtxt(pu_unc_file, skiprows=1)
        T_meth_unc_facs = factor_array[:, 1]
        A_weight_unc_facs = factor_array[:, 2]
    else:
        T_meth_unc_facs = []
        A_weight_unc_facs = []

    return T_meth_unc_facs, A_weight_unc_facs


def load_raw_and_analyzed_files(config, conf_dir):
    ''' common to all systematic sweeps:
    load the raw input file and the associated analysis output file'''

    # load the raw file
    rawf_name = f'{conf_dir}/{config["raw_file"]}'
    raw_f = r.TFile(rawf_name)

    # load the info from the analysis config file
    with open(f'{conf_dir}/{config["fit_conf"]}') as file:
        fit_conf = json.load(file)

    ana_fname = f'{conf_dir}/../{fit_conf["out_dir"]}/'\
        f'{fit_conf["outfile_name"]}'
    if not ana_fname.endswith('.root'):
        ana_fname += '.root'

    return raw_f, load_fit_conf(fit_conf, r.TFile(ana_fname))


def load_fit_conf(fit_conf, ana_file):
    ''' load example fits, pileup uncertainties,
    threshold bin, a_models, etc from the fit_conf'''
    thresh_bin = fit_conf['thresh_bin']

    example_T_hist = ana_file.Get('T-Method/tMethodHist')
    example_fit = example_T_hist.GetFunction('tMethodFit')

    example_A_hist = ana_file.Get('A-Weighted/aWeightHist')
    example_A_fit = example_A_hist.GetFunction('aWeightFit')
    a_model = ana_file.Get('A-Weighted/aVsESpline')

    fit_start, fit_end = example_fit.GetXmin(), example_fit.GetXmax()

    prepare_loss_hist(fit_conf, example_T_hist)

    T_meth_unc_facs, A_weight_unc_facs = load_pu_uncertainties(
        fit_conf.get('pu_uncertainty_file'))

    output_dict = {
        'thresh_bin': thresh_bin,
        'example_T_hist': example_T_hist,
        'example_T_fit': example_fit,
        'T_meth_unc_facs': T_meth_unc_facs,
        'example_A_hist': example_A_hist,
        'example_A_fit': example_A_fit,
        'A_weight_unc_facs': A_weight_unc_facs,
        'a_model': a_model,
        'fit_start': fit_start,
        'fit_end': fit_end,
        'root_file': ana_file,
    }

    return output_dict


def make_output_dir(out_f, outs_T, outs_A, x_vals, dir_name, sweep_par_name):
    ''' store the output of a systematic scan in the output root file'''
    super_dir = out_f.mkdir(dir_name)

    sub_names = ['T-Method', 'A-Weighted']
    for outs, sub_name in zip([outs_T, outs_A], sub_names):
        this_dir = super_dir.mkdir(sub_name)
        this_dir.cd()

        for hist, fit, _ in outs:
            hist.Write()
            fit.Write()

        chi2_g = r.TGraph()
        par_gs = {}

        chi2_g.SetName('chi2')

        for x_val, (_, fit, _) in zip(x_vals, outs):
            pt_num = chi2_g.GetN()

            chi2_g.SetPoint(pt_num, x_val, fit.GetChisquare())
            update_par_graphs(x_val, fit, par_gs)

        g_dir = this_dir.mkdir('sweepGraphs')
        g_dir.cd()

        chi2_g.GetXaxis().SetTitle(sweep_par_name)
        chi2_g.Write()

        for g_name in par_gs:
            par_gs[g_name].GetXaxis().SetTitle(sweep_par_name)
            par_gs[g_name].Write()


#
# Driver function
#

def run_systematic_sweeps(conf_name):
    with open(conf_name) as file:
        config = json.load(file)

    config_dir = '/'.join(conf_name.split('/')[:-1])

    # pileup phase sweep
    pu_phase_conf = config.get('pileup_phase_scan')
    pu_sweep_out = None
    if pu_phase_conf is not None and pu_phase_conf['run_scan']:
        pu_sweep_out = pileup_phase_sweep(pu_phase_conf, config_dir)
        print('\n---\nPileup phase sweep done\n---\n')

    # ifg amplitude sweep
    ifg_amp_conf = config.get('ifg_amplitude_scan')
    ifg_amp_out = None
    if ifg_amp_conf is not None and ifg_amp_conf['run_scan']:
        ifg_amp_out = ifg_amplitude_sweep(ifg_amp_conf, config_dir)
        print('\n---\nIFG amplitude sweep done\n---\n')

    # make output file
    outf_name = config['outfile_name']
    if not outf_name.endswith('.root'):
        outf_name += '.root'

    outf = r.TFile(outf_name, 'recreate')
    print('\n---\nMaking output file\n---\n')
    if pu_sweep_out is not None:
        make_output_dir(outf, *pu_sweep_out, 'pileupPhaseSweep',
                        'pileup time shift [#mus]')
    if ifg_amp_out is not None:
        make_output_dir(outf, *ifg_amp_out, 'ifgAmpSweep',
                        'ifg amplitude multiplier')
