# omega_a analysis functions
#
# also contains the run_analysis driver function
#
# Aaron Fienberg
# September 2018

import math
import subprocess
import ROOT as r
from BlindersPy3 import Blinders, FitType
from .util import *
from .fitmodels import *

r.gStyle.SetOptStat(0)
r.gStyle.SetOptFit(1111)

calo_dirname = 'perCaloPlots'
e_sweep_dirname = 'energyBinnedPlots'

# approximate omega_a period
approx_oma_period = 4.37


def do_threshold_sweep(all_calo_2d, fit_function, fit_start, fit_end,
                       start_thresh=1800):
    ''' finds the optimal T-method threshold
        returns:
        (best_threshold, [(thresh1, r_precision1, thresh2, preceision2),...])

        thresholds are in units of bin number from all_calo_2d's y axis
    '''

    thresh_bin = all_calo_2d.GetYaxis().FindBin(start_thresh)

    r_precisions = []
    best_thresh = None
    for e_bin in range(thresh_bin - 50, thresh_bin + 50):
        this_proj = all_calo_2d.ProjectionX(f'proj{e_bin}', e_bin, -1)

        if fit_function.GetParameter(0) == 0:
            fit_function.SetParameter(
                0, this_proj.GetBinContent(this_proj.FindBin(30)) * 1.6)

        this_proj.Fit(fit_function, '0qEM', '', fit_start, fit_end)

        r_precisions.append((e_bin,
                             fit_function.GetParError(4)))

        if best_thresh is None or best_thresh[-1] > r_precisions[-1][-1]:
            best_thresh = r_precisions[-1]

    return best_thresh[0], r_precisions


def find_cbo_freq(fft_hist):
    return fft_hist.GetBinCenter(fft_hist.GetMaximumBin())


def fit_and_fft(hist, func, fit_name, fit_options,
                fit_start, fit_end, find_cbo=False):
    ''' fit hist to a function and FFT the residuals
        returns the residuals hist and the FFT hist

        if find_cbo is true,
        cbo_freq will be estimated from the largest FFT peak
        and returned after the FFT hist as a third return value
    '''

    # fit twice, often finds a better minimum the second time
    hist.Fit(func, fit_options, '', fit_start, fit_end)

    if adjust_phase_parameters(func):
        # adjusted some phase parameters, try fitting again
        hist.Fit(func, fit_options, '', fit_start, fit_end)

    resids = build_residuals_hist(hist, func.GetName())
    resid_fft = fft_histogram(after_t(resids, fit_start), f'{fit_name}FFT')
    resid_fft.SetTitle(f'{hist.GetTitle()};f [MHz];fft mag')

    retvals = (resids, resid_fft)

    if find_cbo:
        cbo_freq = find_cbo_freq(resid_fft)
        retvals = retvals + (cbo_freq,)

    return retvals


def fit_slice(master_3d, name, model_fit,
              fit_options, fit_range, energy_bins, calo_bins,
              adjust_N=False):
    ''' do a projected T-method fit
        starting fit_guesses are based on model_fit
        projects in region defined by energy_bins and calo_bins
        returns
        energy_bins are in bin numbers, not energies

        if adjust_N is true, sets the N0 guess based on
        bin content at 30 microseconds

        returns fit, resids, fft
    '''

    master_3d.GetXaxis().SetRange(0, master_3d.GetNbinsX())
    master_3d.GetYaxis().SetRange(*energy_bins)
    master_3d.GetZaxis().SetRange(*calo_bins)

    hist = master_3d.Project3D('x')
    hist.SetName(name)

    # hist = calo_2d.ProjectionX(f'{name}', energy_bins[0], energy_bins[1])

    bin_width = hist.GetBinWidth(1)
    hist.SetTitle(f';time [#mu s]; N / {bin_width:.3f} #mu s')

    if adjust_N:
        model_fit.SetParameter(0,
                               hist.GetBinContent(hist.FindBin(30)) * 1.6)

    # find a reasonable guess for the asymmetry
    # by comparing max, min, avg over one period
    if is_free_param(model_fit, 2):
        start_bin = hist.FindBin(fit_range[0])
        bins_per_period = int(approx_oma_period / hist.GetBinWidth(1))
        vals = [hist.GetBinContent(i)
                for i in range(start_bin, start_bin + bins_per_period)]
        a_guess = (max(vals) - min(vals)) * len(vals) / sum(vals) / 2
        model_fit.SetParameter(2, a_guess)

    resids, fft = fit_and_fft(
        hist, model_fit, name, fit_options, fit_range[0], fit_range[1])

    return hist, resids, fft


def T_method_analysis(all_calo_2d, blinder, config):
    ''' do a full calo T-method analysis
    returns T-method hist, fit function, TFitResult, threshold_bin
    '''

    # where to put the plots
    pdf_dir = f'{config["out_dir"]}/plots'

    # start with a simple all calo analysis

    # sweep energy threshold to find the best energy cut
    omega_a_ref = blinder.paramToFreq(0)
    five_param_tf1 = build_5_param_func()
    five_param_tf1.SetParameters(0, 64.4, 0.2, 0.2, 0, omega_a_ref)
    five_param_tf1.FixParameter(5, omega_a_ref)
    print('doing intial threshold sweep...')

    optimal_thresh_bin, sweep_res = do_threshold_sweep(
        all_calo_2d, five_param_tf1, config['fit_start'], config['fit_end'])
    c, _ = plot_threshold_sweep(all_calo_2d, sweep_res, optimal_thresh_bin)
    c.Print(f'{pdf_dir}/optimalThreshold.pdf')

    print(f'best threshold bin: {optimal_thresh_bin}')
    best_thresh = all_calo_2d.GetYaxis().GetBinCenter(optimal_thresh_bin)
    print(f'best threshold: {best_thresh}')
    print('')

    # now do a five parameter fit with the optimal threshold
    best_T_hist = all_calo_2d.ProjectionX(
        'tMethodHist', optimal_thresh_bin, -1)
    five_param_tf1.SetParameter(0,
                                best_T_hist.GetBinContent(
                                    best_T_hist.FindBin(30)) * 1.6)
    bin_width = best_T_hist.GetBinWidth(1)
    best_T_hist.SetTitle(
        f'T-method, all calos, {best_thresh/1000:.2f} GeV threshold; ' +
        f'time [#mu s]; N / {bin_width:.3f} #mu s')

    print('five parameter fit to T method histogram...')

    resids, fft, cbo_freq = fit_and_fft(
        best_T_hist, five_param_tf1, 'fiveParamAllCalos',
        config['fit_options'],
        config['fit_start'], config['fit_end'], True)
    print_fit_plots(best_T_hist, fft, cbo_freq, 'fiveParamAllCalos', pdf_dir)

    print('')
    print(f'estimated CBO frequency is {cbo_freq:.2f} MHz')
    print(f'this correspons to n = {n_of_CBO_freq(cbo_freq):.3f}')

    print('\nfitting with cbo N term...')
    with_cbo_tf1 = build_CBO_only_func(five_param_tf1, cbo_freq)

    resids, fft = fit_and_fft(
        best_T_hist, with_cbo_tf1, 'cboFitAllCalos',
        config['fit_options'],
        config['fit_start'], config['fit_end'])
    print_fit_plots(best_T_hist, fft, cbo_freq, 'cboFitAllCalos', pdf_dir)

    cbo_freq = with_cbo_tf1.GetParameter(9) / 2 / math.pi

    print(f'\nfitted CBO frequency is {cbo_freq: .3f} MHz')
    print(f'this correspons to n = {n_of_CBO_freq(cbo_freq):.3f}')

    print('\nfitting with VW N term...')
    vw_tf1 = build_CBO_VW_func(with_cbo_tf1, cbo_freq)

    resids, fft = fit_and_fft(
        best_T_hist, vw_tf1, 'vwFitAllCalos',
        config['fit_options'],
        config['fit_start'], config['fit_end'])
    print_fit_plots(best_T_hist, fft, cbo_freq, 'vwFitAllCalos', pdf_dir)

    print('\npreparing muon loss histograms...')
    muon_hists = prepare_loss_hist(config, best_T_hist)
    c, _ = plot_loss_hists(*muon_hists)
    c.Print(f'{pdf_dir}/lostMuonPlot.pdf')

    print('\nfitting with muon loss term included...')
    loss_tf1 = build_losses_func(vw_tf1)

    resids, fft = fit_and_fft(
        best_T_hist, loss_tf1, 'lossFitAllCalos',
        config['fit_options'],
        config['fit_start'], config['fit_end'])
    print_fit_plots(best_T_hist, fft, cbo_freq, 'lossFitAllCalos', pdf_dir)

    print('\afitting with full model over full range...')

    full_fit_tf1 = build_full_fit_tf1(loss_tf1, config)
    full_fit_tf1.SetName('tMethodFit')
    resids, fft = fit_and_fft(
        best_T_hist, full_fit_tf1, 'fullFitAllCalos',
        config['fit_options'],
        config['fit_start'], config['extended_fit_end'])
    print_fit_plots(best_T_hist, fft, cbo_freq, 'fullFitAllCalos', pdf_dir)

    # grab the covariance matrix from the full fit
    fr = best_T_hist.Fit(full_fit_tf1, config['fit_options'] + 'S', '',
                         config['fit_start'], config['extended_fit_end'])

    corr_mat = fr.GetCorrelationMatrix()
    print('')
    print('parameter correlations with R: ')
    for i in range(full_fit_tf1.GetNpar()):
        corr = corr_mat(i, 4)
        if corr != 0:
            print(f'R-{full_fit_tf1.GetParName(i)} correlation: ' +
                  f'{corr:.2f}')

    print('\nfinished basic T-method analysis\n')

    return best_T_hist, full_fit_tf1, fr, optimal_thresh_bin


def T_method_calo_sweep(master_3d, model_fit, thresh_bin, config):
    ''' sweep over all calos, fitting each one
    returns a list of (hist, model_fit, resids, fft)) for each calo
    uses model_fit to determine guesses for first calo
    after that, it uses the results from the previous calo
    '''

    print('T method calo sweep...')
    results = []
    for i in range(1, 25):
        model_fit = clone_full_fit_tf1(model_fit, f'Calo{i}TFit')

        print(f'calo {i}')

        # adjust N only for the first calo
        adjust_N = True if i == 1 else False

        hist, resids, fft = fit_slice(master_3d, f'Calo{i}THist', model_fit,
                                      fit_options=config['fit_options'],
                                      fit_range=(config['fit_start'],
                                                 config['fit_end']),
                                      energy_bins=(
                                          thresh_bin, master_3d.GetNbinsY()),
                                      calo_bins=(i, i),
                                      adjust_N=adjust_N)

        hist.SetTitle(f'calo {i} T-method')
        fft.SetTitle(f'calo {i} T-method')

        results.append((hist, model_fit, resids, fft))

    return results


def energy_sweep(master_3d, model_fit, min_e, max_e, n_slices, config):
    ''' sweep over energy bins, fitting each one
    returns a list of (hist, [e_low, e_high], model_fit, resids, fft))
    for each energy
    uses model_fit as starting guess for the first energy bin
    after that, it uses the results from the previous bin

    n_slices is approximate, the actual number will be such that
    each histogram has the same energy width,
    and all histograms span the range from min_e to max_e
    '''

    print('energy binned sweep...')
    results = []

    # convert energy range into bin index ranges
    energy_axis = master_3d.GetYaxis()

    low_bin = energy_axis.FindBin(min_e)
    high_bin = energy_axis.FindBin(max_e) + 1

    bins_per_slice = math.ceil((high_bin - low_bin) / n_slices)
    bin_ranges = [(start, start + bins_per_slice - 1)
                  for start in range(low_bin, high_bin, bins_per_slice)]

    for i, (low_bin, high_bin) in enumerate(bin_ranges):
        low_e = energy_axis.GetBinLowEdge(low_bin)
        high_e = energy_axis.GetBinLowEdge(high_bin + 1)
        avg_e = 0.5 * (low_e + high_e)

        model_fit = clone_full_fit_tf1(model_fit,
                                       f'eFit{avg_e:.0f}')

        print(f'slice from {low_e:.0f} to {high_e:.0f} MeV')

        # # adjust N only for the first calo
        adjust_N = True if i == 0 else False

        hist, resids, fft = \
            fit_slice(master_3d, f'eBin{avg_e:.0f}', model_fit,
                      fit_options=config['fit_options'],
                      fit_range=(config['fit_start'],
                                 config['fit_end']),
                      energy_bins=(low_bin, high_bin),
                      calo_bins=(1, master_3d.GetNbinsZ()),
                      adjust_N=adjust_N)

        hist.SetTitle(f'all calos, {low_e:.0f} to {high_e:.0f} MeV')
        fft.SetTitle(f'all calos, {low_e:.0f} to {high_e:.0f} MeV')

        results.append((hist, (low_e, high_e), model_fit, resids, fft))

    return results


def T_meth_pu_mult_scan(corrected_2d, uncorrected_2d,
                        thresh_bin, model_fit, scales, config):
    ''' vary the pileup multiplier and fit

    corrected_2d: pileup corrected 2d histogram
    uncorrected_2d: pileup uncorrected 2d histogram
    thresh_bin: T-method threshold energy bin
    model_fit: fit function to use, should be a "full fit"
    scales: a list of scale factors to use
    config: analysis config dictionary

    returns list of fit functions at each scale factor
    '''

    uncorrected_T_hist = uncorrected_2d.ProjectionX(
        'uncorrectedT', thresh_bin, -1)

    # pu perturbation is uncorrected T hist minus corrected T hist
    pu_pert = uncorrected_T_hist.Clone()
    pu_pert.SetName('puPertTMeth')
    # corrected_T_hist =
    pu_pert.Add(corrected_2d.ProjectionX('correctedTMeth',
                                         thresh_bin, -1), -1)

    pu_scan_fit = clone_full_fit_tf1(model_fit,
                                     f'tFitScaled{scales[0]}')

    fits = []
    for scale_factor in scales:
        fit_name = f'tFitScaled{scale_factor}'
        if len(fits) == 0:
            fit = pu_scan_fit
        else:
            fit = clone_full_fit_tf1(fits[-1], fit_name)

        fit_hist = uncorrected_T_hist.Clone()
        fit_hist.Add(pu_pert, -1 * scale_factor)

        fit_hist.ResetStats()
        fit_hist.Fit(fit, config['fit_options'] + '0', ' ',
                     config['fit_start'], config['extended_fit_end'])

        fits.append(fit)

    return fits


def make_pu_scan_graphs(scales, fits):
    ''' returns chi_2 vs scan_x
    and also parameter_graphs dictionary for all
    non fixed fit parameters '''

    chi2_g = r.TGraph()

    par_gs = {}

    for scale, fit in zip(scales, fits):
        chi2_g.SetPoint(chi2_g.GetN(), scale,
                        fit.GetChisquare())

        update_par_graphs(scale, fit, par_gs)

    return chi2_g, par_gs


def make_calo_sweep_graphs(calo_sweep_res):
    ''' returns chi2_g, par_gs '''
    chi2_g = r.TGraphErrors()
    chi2_g.SetName('calo_chi2_g')
    chi2_g.SetTitle(';calo num; #chi^{2}/ndf')

    par_gs = {}

    for calo_num, (hist, fit, _, fft) in \
            enumerate(calo_sweep_res, 1):

        pt_num = calo_num - 1

        chi2_g.SetPoint(pt_num, calo_num,
                        fit.GetChisquare() / fit.GetNDF())
        chi2_g.SetPointError(pt_num, 0, math.sqrt(2 / fit.GetNDF()))

        update_par_graphs(calo_num, fit, par_gs)

    return chi2_g, par_gs


def make_E_sweep_graphs(energy_sweep_res):
    ''' returns chi2_g, par_gs '''
    chi2_g = r.TGraphErrors()
    chi2_g.SetName('energy_chi2_g')
    chi2_g.SetTitle(';energy [MeV]; #chi^{2}/ndf')

    par_gs = {}

    for pt_num, (hist, (low_e, high_e), fit, _, fft) in \
            enumerate(energy_sweep_res):

        energy = 0.5 * (low_e + high_e)

        chi2_g.SetPoint(pt_num, energy,
                        fit.GetChisquare() / fit.GetNDF())
        chi2_g.SetPointError(pt_num, 0, math.sqrt(2 / fit.GetNDF()))

        update_par_graphs(energy, fit, par_gs)

    return chi2_g, par_gs


#
# Stuff for A-Weighted analysis
#


def build_A_vs_E_spline(a_vs_e, phi_vs_e, deriv_cut=0.01):
    '''
    builds a signed A vs E spline out of
    an unsigned a_vs_e graph and a phi_vs_e graph
    returns a_vs_e_spline, signed_a_vs_e_graph
    '''

    # find energy at which asymmetry changes sign
    # base this on a large slope of phi versus E
    x, y = r.Double(), r.Double()
    phi_vs_e.GetPoint(0, x, y)
    last_x, last_y = float(x), float(y)
    inversion_e = 0
    for pt_num in range(1, phi_vs_e.GetN()):
        phi_vs_e.GetPoint(pt_num, x, y)
        abs_deriv = abs((y - last_y) / (x - last_x))
        if abs_deriv > deriv_cut:
            inversion_e = float(x)
            break
        last_x, last_y = float(x), float(y)

    # build signed version of A versus E
    signed_a_vs_e = r.TGraph()
    for pt_num in range(a_vs_e.GetN()):
        a_vs_e.GetPoint(pt_num, x, y)
        adjusted_y = y
        if x < inversion_e:
            adjusted_y *= -1

        signed_a_vs_e.SetPoint(pt_num, x, adjusted_y)

    # build an interpolating spline
    a_vs_e_spline = r.TSpline3('A vs E spline', signed_a_vs_e)
    a_vs_e_spline.SetName('aVsESpline')

    signed_a_vs_e.SetName('aVsEGraph')
    signed_a_vs_e.SetTitle(';energy [MeV]; A')

    return signed_a_vs_e, a_vs_e_spline


def build_a_weight_hist(spec_2d, a_vs_e_spline, name,
                        min_e=1000, max_e=3000):
    ''' builds an asymmetry weighted histogram,
    pretty self explanatory
    '''

    # don't go above the max energy available from the a_vs_e_spline
    if max_e > a_vs_e_spline.GetXmax():
        max_e = a_vs_e_spline.GetXmax()

    time_axis = spec_2d.GetXaxis()
    bin_width = time_axis.GetBinWidth(1)
    a_weight_hist = r.TH1D(name, 'A-Weighted;' +
                           f' t [#mus]; N / {bin_width:.3f} #mus',
                           time_axis.GetNbins(),
                           time_axis.GetBinLowEdge(1),
                           time_axis.GetBinUpEdge(time_axis.GetNbins()))
    a_weight_hist.Sumw2()

    start_bin = spec_2d.GetYaxis().FindBin(min_e)
    end_bin = spec_2d.GetYaxis().FindBin(max_e) + 1

    for e_bin in range(start_bin, end_bin):
        energy_slice = spec_2d.ProjectionX(f'e_slice{e_bin}', e_bin, e_bin)
        energy = spec_2d.GetYaxis().GetBinCenter(e_bin)
        a_weight_hist.Add(energy_slice, a_vs_e_spline.Eval(energy))

    return a_weight_hist


def A_weight_pu_mult_scan(corrected_2d, uncorrected_2d,
                          a_vs_e_spline,
                          model_fit, scales, config,
                          min_e=1000, max_e=3000):
    ''' vary the pileup multiplier and fit, for A_Weighted analysis

    corrected_2d: pileup corrected 2d histogram
    uncorrected_2d: pileup uncorrected 2d histogram
    a_vs_e_spline: asymmetry versus energy model
    model_fit: fit function to use, should be a "full fit"
    scales: a list of scale factors to use
    config: analysis config dictionary
    returns list of fit functions at each scale factor
    min_e: minimum energy to include
    max_e: maximum energy to include
    '''

    # 2d pileup perturbation is uncorrected minus corrected
    pu_pert_2d = uncorrected_2d.Clone()
    pu_pert_2d.SetName('2d_pu_pert')
    pu_pert_2d.Add(corrected_2d, -1)

    pu_scan_fit = clone_full_fit_tf1(model_fit,
                                     f'aWeightScaled{scales[0]}')

    fits = []
    for scale_factor in scales:
        fit_name = f'aWeightFitScaled{scale_factor}'
        if len(fits) == 0:
            fit = pu_scan_fit
        else:
            fit = clone_full_fit_tf1(fits[-1], fit_name)

        scaled_2d = uncorrected_2d.Clone()
        scaled_2d.Add(pu_pert_2d, -1 * scale_factor)

        scaled_a = build_a_weight_hist(scaled_2d, a_vs_e_spline,
                                       f'scaled_aweight_{scale_factor}',
                                       min_e=min_e,
                                       max_e=max_e)

        scaled_a.Fit(fit, config['fit_options'] + '0', ' ',
                     config['fit_start'], config['extended_fit_end'])

        fits.append(fit)

    return fits

#
# plotting and printing functions
#
# print functions make pdf plots
# plot functions return a canvas (and newly created objects in the canvas)
#


def plot_hist(hist):
    c = r.TCanvas()
    hist.Draw()
    c.SetLogy(1)

    return c, []


def plot_fft(fft, cbo_freq):
    c = r.TCanvas()

    fft.Draw()
    lns = plot_expected_freqs(fft, cbo_freq)

    return c, lns


def print_fit_plots(hist, fft, cbo_freq, fit_name, pdf_dir):
    c, _ = plot_hist(hist)
    c.Draw()

    c.Print(f'{pdf_dir}/{fit_name}.pdf')

    c, _ = plot_fft(fft, cbo_freq)
    c.Print(f'{pdf_dir}/{fit_name}FFT.pdf')


def plot_expected_freqs(resid_fft, cbo_freq, f_c=1.0 / 0.149, f_a=0.2291):
    y_min, y_max = 0, resid_fft.GetMaximum() * 1.1
    resid_fft.GetYaxis().SetRangeUser(y_min, y_max)

    n = n_of_CBO_freq(cbo_freq)

    expected_freqs = {'f_{cbo}': cbo_freq, 'f_{y}': math.sqrt(n) * f_c,
                      'f_{vw}': f_c * (1 - 2 * math.sqrt(n)),
                      'f_{a} - f_{cbo}': cbo_freq - f_a,
                      'f_{a} + f_{cbo}': cbo_freq + f_a,
                      '2*f_{cbo}': cbo_freq * 2}

    lns = []
    for name in expected_freqs:
        freq = expected_freqs[name]
        ln = r.TLine(freq, y_min, freq, y_max)
        ln.SetLineStyle(2)
        ln.SetLineWidth(2)
        ln.Draw()
        lns.append(ln)

    return lns


def plot_threshold_sweep(all_calo_2d, r_precisions, best_bin):
    # make a plot of the threshold sweep result
    precision_vs_threshg = r.TGraph()
    for i, (e_bin, precision) in enumerate(r_precisions):
        precision_vs_threshg.SetPoint(i,
                                      all_calo_2d.GetYaxis().
                                      GetBinCenter(e_bin),
                                      precision)

    best_Ecut = all_calo_2d.GetYaxis().GetBinCenter(best_bin)
    precision_vs_threshg.SetTitle(f'optimal threshold: {best_Ecut} MeV;'
                                  'energy threshold [MeV];'
                                  ' #omega_{a} uncertainty [ppm]')

    c = r.TCanvas()
    precision_vs_threshg.Draw('ap')
    y_min, y_max = min(precision_vs_threshg.GetY()) * \
        0.9, max(precision_vs_threshg.GetY()) * 1.1
    precision_vs_threshg.GetYaxis().SetRangeUser(y_min, y_max)

    line = r.TLine(best_Ecut, y_min, best_Ecut, y_max)
    line.SetLineStyle(2)
    line.SetLineWidth(2)
    line.Draw()
    c.SetLogy(0)

    return c, [precision_vs_threshg, line]


def plot_loss_hists(lost_muon_rate,
                    lost_muon_prob):
    # scale histograms for plotting
    lost_muon_rate.Scale(1.0 / lost_muon_rate.Integral('width'))
    lost_muon_prob.Scale(1.0 / lost_muon_prob.Integral('width'))
    plot_cumulative = lost_muon_prob.GetCumulative()
    plot_cumulative.Scale(
        lost_muon_rate.GetMaximum() / plot_cumulative.GetMaximum())

    c = r.TCanvas()
    lost_muon_rate.Draw('hist')
    lost_muon_rate.SetTitle('')
    lost_muon_rate.GetXaxis().SetRangeUser(0, 300)
    lost_muon_rate.GetYaxis().SetRangeUser(1e-6, 1)

    lost_muon_prob.SetLineColor(r.kRed)
    lost_muon_prob.Draw('hist same')
    plot_cumulative.SetLineColor(r.kBlue)
    plot_cumulative.Draw('hist same')
    c.SetLogy(1)
    c.Draw()
    leg = r.TLegend(0.5, 0.8, 0.8, 0.5)
    leg.SetLineColor(r.kWhite)
    leg.SetFillColor(r.kWhite)
    leg.SetShadowColor(r.kWhite)
    leg.AddEntry(lost_muon_rate, 'L(t)', 'l')
    leg.AddEntry(lost_muon_prob, 'L(t) #bullet exp(t / 64.4 #mu s)', 'l')
    leg.AddEntry(plot_cumulative, '#int_{0}^{t} L(t\')'
                 ' #bullet exp(t\' / 64.4 #mu s) dt\'', 'l')

    leg.Draw()

    return c, [plot_cumulative, leg]


def print_calo_sweep_res(calo_sweep_res, chi2_g, par_gs, pdf_dir):
    print('making calorimeter sweep plots')

    r.gStyle.SetStatH(0.15)

    for calo_num, (hist, fit, _, fft) in \
            enumerate(calo_sweep_res, 1):

        cbo_freq = fit.GetParameter(9) / 2 / math.pi
        print_fit_plots(hist, fft, cbo_freq,
                        f'{calo_dirname}/calo{calo_num}Tfit', pdf_dir)

    c = r.TCanvas()
    chi2_g.Draw('ap')
    chi2_g.GetXaxis().SetLimits(0, 25)
    ln = r.TLine(0, 1, 25, 1)
    ln.SetLineWidth(2)
    ln.SetLineStyle(2)
    ln.Draw()

    c.Print(f'{pdf_dir}/{calo_dirname}/chi2VsCalo.pdf')

    for par_num in par_gs:
        g = par_gs[par_num]
        g.GetXaxis().SetTitle('calo num')

        g.Draw('ap')

        par_name = g.GetYaxis().GetTitle()
        par_name = strip_par_name(par_name)

        g.GetXaxis().SetLimits(0, 25)

        c.Print(f'{pdf_dir}/{calo_dirname}/{par_name}VsCalo.pdf')


def print_energy_sweep_res(energy_sweep_res, chi2_g, par_gs, pdf_dir):
    print('making energy sweep plots')

    r.gStyle.SetStatH(0.2)

    for pt_num, (hist, (low_e, high_e), fit, _, fft) in \
            enumerate(energy_sweep_res):

        energy = 0.5 * (low_e + high_e)

        cbo_freq = fit.GetParameter(9) / 2 / math.pi
        print_fit_plots(hist, fft, cbo_freq,
                        f'{e_sweep_dirname}/eFit{energy:.0f}', pdf_dir)

    c = r.TCanvas()
    chi2_g.SetTitle(';energy [MeV]; #chi^{2}/ndf')
    chi2_g.Draw('ap')

    c.Print(f'{pdf_dir}/{e_sweep_dirname}/chi2VsEnergy.pdf')

    for par_num in par_gs:
        g = par_gs[par_num]
        g.GetXaxis().SetTitle('energy [MeV]')

        g.Draw('ap')

        par_name = g.GetYaxis().GetTitle()
        par_name = strip_par_name(par_name)

        c.Print(f'{pdf_dir}/{e_sweep_dirname}/{par_name}VsEnergy.pdf')


def plot_pu_sweep_R(par_gs, title):
    ''' returns canvas
    '''
    # make R plot
    c1 = r.TCanvas()
    r_g = par_gs[4]
    r_g.SetTitle(f'{title};pu multiplier; R')
    r_g.Fit('pol1', 'EMq')
    r_g.Draw('ap E0X')

    return c1


def plot_pu_sweep_chi2(chi2_g, title):
    ''' returns canvas
    '''
    c1 = r.TCanvas()
    chi2_g.SetTitle(f'{title};pu multiplier; #chi^{{2}}')
    chi2_g.Draw()

    # make chi2 plot

    chi2s = list(chi2_g.GetY())
    min_chi2 = min(chi2s)
    min_chi2_mult = chi2_g.GetX()[chi2s.index(min_chi2)]

    chi2_g.Draw('ap')

    chi2_g_fit = r.TF1('pu_chi2_fit', '[1]*(x-[0])^2 + [2]')
    chi2_g_fit.SetParName(0, 'a')
    chi2_g_fit.SetParName(1, 'b')
    chi2_g_fit.SetParName(2, 'c')
    chi2_g_fit.SetParameter(0, min_chi2_mult)
    chi2_g_fit.SetParameter(1, 0)
    chi2_g_fit.SetParameter(2, min_chi2)
    chi2_g.Fit(chi2_g_fit, 'EMq')

    return c1


def plot_a_vs_e_curve(a_vs_e_g, a_vs_e_spline):
    c = r.TCanvas()

    a_vs_e_g.Draw('ap')
    a_vs_e_g.GetXaxis().SetLimits(0, 3000)
    a_vs_e_spline.Draw('same')
    ln = r.TLine(0, 0, 3000, 0)
    ln.SetLineStyle(2)
    ln.Draw()

    return c, ln


#
# driver function to run the analysis
#


def run_analysis(config):
    #
    # build necessary output directories
    #
    out_dir = config['out_dir']
    subprocess.call(f'mkdir -p {out_dir}'.split())
    pdf_dir = f'{out_dir}/plots'
    subprocess.call(f'mkdir -p {pdf_dir}'.split())
    calo_dir = f'{pdf_dir}/{calo_dirname}'
    subprocess.call(f'mkdir -p {calo_dir}'.split())
    e_bin_dir = f'{pdf_dir}/{e_sweep_dirname}'
    subprocess.call(
        f'mkdir -p {e_bin_dir}'.split())

    #
    # setup blinder object and read histograms from input file
    #

    blinder = Blinders(FitType.Omega_a, config['blinding_phrase'])

    uncorrected_3d = get_histogram(config['uncor_hist_name'],
                                   config['file_name'])

    all_calo_2d = master_3d.Project3D('yx_all')

    r.gStyle.SetStatW(0.25)
    r.gStyle.SetStatH(0.4)

    #
    # Start with a T-Method analysis
    #

    T_hist, full_fit, result, thresh = T_method_analysis(
        all_calo_2d, blinder, config)

    print('T-Method pileup multiplier scan...')

    # do T-method pileup multiplier scan
    uncorrected_2d = uncorrected_3d.Project3D('yx')

    pu_scan_fit = clone_full_fit_tf1(full_fit, 'pu_scan_fit')
    pu_scan_fit.SetParLimits(6, 100, 400)

    scale_factors = [i / 10 + 0.4 for i in range(10)]
    pu_scale_fits = T_meth_pu_mult_scan(all_calo_2d, uncorrected_2d,
                                        thresh, pu_scan_fit,
                                        scale_factors, config)
    t_pu_chi2_g, t_pu_par_gs = make_pu_scan_graphs(
        scale_factors, pu_scale_fits)
    c1 = plot_pu_sweep_R(t_pu_par_gs, 'T-Method')
    c1.Print(f'{pdf_dir}/tMethodPuSweepR.pdf')
    c1 = plot_pu_sweep_chi2(t_pu_chi2_g, 'T-Method')
    c1.Print(f'{pdf_dir}/tMethodPuSweepchi2.pdf')

    #
    # Do a per calo analysis
    #

    print('Per calo analysis...')

    per_calo_fit = clone_full_fit_tf1(full_fit, 'per_calo_fit')

    # free 2*omega_cbo params for per calo
    for par_num in range(24, 27):
        per_calo_fit.ReleaseParameter(par_num)

    # limit the cbo lifetime params
    per_calo_fit.SetParLimits(6, 50, 400)
    per_calo_fit.SetParLimits(24, 30, 200)

    # fix vw parameters for the single calo fit
    for par_num in [10, 13]:
        per_calo_fit.FixParameter(par_num,
                                  per_calo_fit.GetParameter(par_num))

    # T-method fits per calo
    calo_sweep_res = T_method_calo_sweep(
        master_3d, per_calo_fit, thresh, config)
    calo_chi2_g, calo_sweep_par_gs = make_calo_sweep_graphs(calo_sweep_res)
    print_calo_sweep_res(calo_sweep_res, calo_chi2_g,
                         calo_sweep_par_gs, pdf_dir)

    print('energy binned analysis...')

    e_binned_fit = clone_full_fit_tf1(full_fit, 'e_binned_fit')

    # fix a number of parameters for the energy binned fits
    for par_num in [6, 9, 10, 13] + list(range(19, 27)):
        e_binned_fit.FixParameter(par_num, e_binned_fit.GetParameter(par_num))

    # do the energy bin sweeps
    e_sweep_res = energy_sweep(master_3d, e_binned_fit, 500, 2850, 32, config)
    e_sweep_chi2_g, e_sweep_par_gs = make_E_sweep_graphs(e_sweep_res)
    print_energy_sweep_res(e_sweep_res, e_sweep_chi2_g,
                           e_sweep_par_gs, pdf_dir)

    print('A-Weighted Analysis...')

    signed_a_vs_e, a_vs_e_spline = build_A_vs_E_spline(
        e_sweep_par_gs[2], e_sweep_par_gs[3])
    c, _ = plot_a_vs_e_curve(signed_a_vs_e, a_vs_e_spline)
    c.Print(f'{pdf_dir}/aVsECurve.pdf')

    a_weight_hist = build_a_weight_hist(
        all_calo_2d, a_vs_e_spline, 'aWeightHist')

    a_weight_fit = clone_full_fit_tf1(full_fit, 'aWeightFit')

    a_weight_fit.SetParLimits(10, 20, 40)
    # fix vw, a-weight fit seems to have issues with it
    a_weight_fit.FixParameter(13, a_weight_fit.GetParameter(13))

    a_weight_fit.SetParameter(0,
                              a_weight_hist.GetBinContent(
                                  a_weight_hist.FindBin(30)) * 1.6)

    resids, fft = fit_and_fft(
        a_weight_hist, a_weight_fit, 'fullFitAWeight',
        config['fit_options'],
        config['fit_start'], config['extended_fit_end'])
    print_fit_plots(a_weight_hist, fft,
                    a_weight_fit.GetParameter(9) / 2 / math.pi,
                    'aWeightedFit', pdf_dir)

    print('A-Weighted pileup multiplier scan...')

    a_pu_fit = clone_full_fit_tf1(a_weight_fit, 'a_pu_fit')
    a_pu_scan_fits = A_weight_pu_mult_scan(all_calo_2d, uncorrected_2d,
                                           a_vs_e_spline, a_pu_fit,
                                           scale_factors, config)

    a_pu_chi2_g, a_pu_par_gs = make_pu_scan_graphs(
        scale_factors, a_pu_scan_fits)
    c1 = plot_pu_sweep_R(a_pu_par_gs, 'A-Weighted')
    c1.Print(f'{pdf_dir}/aWeightPuSweepR.pdf')
    c1 = plot_pu_sweep_chi2(a_pu_chi2_g, 'A-Weighted')
    c1.Print(f'{pdf_dir}/aWeightPuSweepchi2.pdf')

    #
    # make an output file with resulting root objects
    #

    out_f_name = config['outfile_name']
    if not out_f_name.endswith('.root'):
        out_f_name = out_f_name + '.root'
    out_f = r.TFile(f'{out_dir}/{out_f_name}', 'recreate')

    # save A weighted and T method plots
    T_hist.Write()
    full_fit.Write()
    a_weight_hist.Write()
    a_weight_fit.Write()

    # save A vs E model
    signed_a_vs_e.Write()
    a_vs_e_spline.Write()

    # save energy binned plots
    e_sweep_dir = out_f.mkdir('energySweep')
    e_sweep_dir.cd()
    for par_num in e_sweep_par_gs:
        graph = e_sweep_par_gs[par_num]
        graph.SetName(f'par{par_num}VsE')
        graph.Write()
    for result in e_sweep_res:
        result[0].Write()
        result[2].Write()
    e_sweep_chi2_g.Write()

    # save calo sweep plots
    calo_sweep_dir = out_f.mkdir('caloSweep')
    calo_sweep_dir.cd()
    for par_num in calo_sweep_par_gs:
        graph = calo_sweep_par_gs[par_num]
        graph.SetName(f'par{par_num}VsCalo')
        graph.Write()
    for result in calo_sweep_res:
        result[0].Write()
        result[1].Write()
    calo_chi2_g.Write()

    # save pileup multiplier scan results
    pu_scan_dir = out_f.mkdir('pileupScans')
    t_pu_dir = pu_scan_dir.mkdir('tMethod')
    t_pu_dir.cd()
    t_pu_chi2_g.SetName('tMethodChi2VsPuScale')
    t_pu_chi2_g.Write()
    for par_num in t_pu_par_gs:
        graph = t_pu_par_gs[par_num]
        graph.SetName(f'tMethodPar{par_num}VsPuScale')
        graph.Write()
    a_pu_dir = pu_scan_dir.mkdir('aWeighted')
    a_pu_dir.cd()
    a_pu_chi2_g.SetName('aWeightedChi2VsPuScale')
    a_pu_chi2_g.Write()
    for par_num in a_pu_par_gs:
        graph = t_pu_par_gs[par_num]
        graph.SetName(f'aWeightedPar{par_num}VsPuScale')
        graph.Write()

    out_f.Write()

    print('Done!')
