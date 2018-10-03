# utility functions to help with omega_a analysis
#
# Aaron Fienberg
# September 2018

import ROOT as r
import math

# approximate omega_a period
approx_oma_period = 4.37


def rebinned_last_axis(array, rebin_factor):
    ''' rebin ND arrays along the last axis '''
    new_shape = list(array.shape)
    new_shape[-1] //= rebin_factor
    new_shape.append(rebin_factor)

    return array.reshape(new_shape).sum(axis=-1)


def get_histogram(hist_name, root_file_name):
    file = r.TFile(root_file_name)
    hist = file.Get(hist_name)
    hist.SetDirectory(0)
    return hist


def is_free_param(func, par_num):
    ''' check if a parameter is free '''
    min_val = r.Double()
    max_val = r.Double()
    func.GetParLimits(par_num, min_val, max_val)

    return min_val != max_val or min_val == 0 and max_val == 0


def AcAs_to_APhi(func, Ac_num, As_num):
    ''' transforms parameters used in an Ac, As parameterization
    into parameters for an a, phi parameterization'''
    Ac, As = func.GetParameter(Ac_num), func.GetParameter(As_num)

    A = math.sqrt(Ac**2 + As**2)
    phi = math.atan2(As, Ac)

    func.SetParameter(Ac_num, A)
    func.SetParameter(As_num, phi)
    # func.SetParLimits(As_num, -math.pi, math.pi)


def adjust_phase_parameters(func):
    '''puts any phase parameters into the range [-pi, pi]
    also makes all asymmetry terms positive
    returns whether any parameters were adjusted
    '''
    any_adjusted = False

    for par_num in range(func.GetNpar()):
        if not is_free_param(func, par_num):
            # ignore fixed parameters
            continue

        val = func.GetParameter(par_num)

        # if we have negative asymmetry term,
        # flip its sign
        # and shift the associated phase by pi
        #
        # The way I've set it up, asymmetry parameters
        # come before associated phase parameters
        if func.GetParName(par_num + 1).startswith('#phi'):
            if val < 0:
                any_adjusted = True
                func.SetParameter(par_num, -1 * val)

                # shift assocaited phase by pi
                phase_val = func.GetParameter(par_num + 1)
                func.SetParameter(par_num + 1, phase_val - math.pi)

        if not func.GetParName(par_num).startswith('#phi'):
            # not a phase parameter, move on to next one
            continue

        while not -math.pi <= val < math.pi:
            any_adjusted = True
            if val < -math.pi:
                val += 2 * math.pi
            else:
                val -= 2 * math.pi

        func.SetParameter(par_num, val)

    return any_adjusted


def strip_par_name(par_name):
    ''' strips '#', {', '\'', '/', ' ', ',' \
    chars out of par_name to make easier to read
    file names '''
    par_name = par_name.replace('#', '')
    par_name = par_name.replace('{', '')
    par_name = par_name.replace('}', '')
    par_name = par_name.replace('/', '')
    par_name = par_name.replace(',', '')
    par_name = par_name.replace(' ', '')

    return par_name


def fft_histogram(histogram, fft_name):
    fft_hist = r.TH1D(fft_name, fft_name,
                      histogram.GetNbinsX(), 0, 1.0 / histogram.GetBinWidth(1))
    histogram.FFT(fft_hist, 'MAG')
    fft_hist.GetXaxis().SetRange(1, fft_hist.GetNbinsX() // 2)
    return fft_hist


def after_t(histogram, t_start):
    ''' cuts out first t_start microseconds of the passed in histogram
    useful for FFTing as we only really care about the FFT
    after t_start microseconds'''
    first_bin = histogram.FindBin(t_start)
    last_bin = histogram.GetNbinsX()
    new_hist = r.TH1D(histogram.GetName() + f'_after{t_start}usec', '',
                      last_bin + 1 - first_bin,
                      histogram.GetBinLowEdge(first_bin),
                      histogram.GetBinLowEdge(last_bin + 1))

    for t_bin in range(first_bin, last_bin + 1):
        new_hist.SetBinContent(t_bin + 1 - first_bin,
                               histogram.GetBinContent(t_bin))

    return new_hist


def build_residuals_hist(histogram, func, use_errors=False, name=None):
    fit_name = func.GetName()

    if name is None:
        name = histogram.GetName() + f'{fit_name}_residuals'

    resid_hist = r.TH1D(name,
                        histogram.GetTitle(),
                        histogram.GetNbinsX(), histogram.GetBinLowEdge(1),
                        histogram.GetBinLowEdge(histogram.GetNbinsX()) + 1)

    for i_bin in range(1, resid_hist.GetNbinsX() + 1):
        content = histogram.GetBinContent(i_bin)
        func_val = func.Eval(histogram.GetBinCenter(i_bin))
        error = histogram.GetBinError(i_bin)
        scale = error if use_errors else 1
        try:
            resid_hist.SetBinContent(i_bin, (content - func_val) / scale)
            resid_hist.SetBinError(i_bin, error / scale)
        except ZeroDivisionError:
            resid_hist.SetBinContent(i_bin, 0)

    return resid_hist


def n_of_CBO_freq(cbo_freq):
    ''' get quad n from cbo frequency
        assumes 149 ns cyclotron period
    '''
    f_c = 1.0 / 0.149
    return (2 * f_c * cbo_freq - cbo_freq**2) / f_c**2


def update_par_graphs(x_val, fit, par_gs):
    '''for per-calo and energy-binned fits,
    it's useful to plot fit parameters versus calo or
    versus E-bin.

    This is a utility function to help with making those plots.

    par_gs starts out as an empty dictionary
    then, looping over calo or e-binned fits,
    one calls update_par_graphs(calo_num/energy, fit, par_gs)
    for each fit (a TF1), and, by the end, par_gs contains
    the desired TGraphErrors for each parameter as a function
    of calo number or energy
    '''
    for par_num in range(fit.GetNpar()):
        if not is_free_param(fit, par_num):
            continue

        try:
            g = par_gs[par_num]
        except KeyError:
            g = r.TGraphErrors()
            g.GetYaxis().SetTitle(fit.GetParName(par_num))
            g.SetName(f'par{par_num}')
            par_gs[par_num] = g

        pt_num = g.GetN()

        y_val = fit.GetParameter(par_num)
        y_err = fit.GetParError(par_num)

        g.SetPoint(pt_num, x_val, y_val)
        g.SetPointError(pt_num, 0, y_err)


# stores param and errors in TGraphErrors
# also stores graphs for low bounds and high bounds
# of allowed statistical drift
class ParamTimeScanResult():
    def __init__(self, func, par_num):
        self._par_num = par_num

        self._g = r.TGraphErrors()

        self._start_var = func.GetParError(par_num)**2
        self._start_val = func.GetParameter(par_num)
        self._g.SetTitle(';start time [#mu s]; {}'.format(
            func.GetParName(par_num)))

        self._low = r.TGraph()
        self._low.SetLineStyle(2)
        self._high = r.TGraph()
        self._high.SetLineStyle(2)

    @property
    def par_num(self):
        return self._par_num

    def add_point(self, time, func):
        point_num = self._g.GetN()
        self._g.SetPoint(point_num,
                         time,
                         func.GetParameter(self._par_num))

        error = func.GetParError(self._par_num)
        self._g.SetPointError(point_num,
                              0,
                              error)

        var = error**2
        var_drift = var - self._start_var
        drift = math.sqrt(var_drift) if var_drift > 0 else 0

        self._low.SetPoint(point_num, time,
                           self._start_val - drift)

        self._high.SetPoint(point_num, time,
                            self._start_val + drift)

    def Draw(self):
        self._g.Draw('ap')
        self._low.Draw('l same')
        self._high.Draw('l same')


def start_time_scan(hist, func, start, step, n_pts,
                    end=None, fit_options=''):
    '''returns one ParamTimeScanResult per parameter'''
    if end is None:
        end = hist.GetBinLowEdge(hist.GetNbinsX() + 1)

    hist.Fit(func, fit_options + '0q', '', start, end)

    results = [ParamTimeScanResult(func, i)
               for i in range(func.GetNpar())
               if is_free_param(func, i)]

    step_in_bins = int(step // hist.GetBinWidth(1))

    start_bin = hist.FindBin(start)

    for i_bin in range(start_bin,
                       start_bin + step_in_bins * n_pts,
                       step_in_bins):
        start = hist.GetBinLowEdge(i_bin)

        hist.Fit(func, fit_options + '0q', '', start, end)

        for result in results:
            result.add_point(start, func)

    return results


def make_shifted_wiggle_func(fit):
    ''' used in the wrapped wiggle plot'''
    def shifted_wiggle(x, p):
        return fit.Eval(x[0] - p[0])
    return shifted_wiggle


def make_wrapped_wiggle_plot(hist, fit, title,
                             fit_start, fit_end,
                             periods_per_wrap=20,
                             earliest_time=3):
    ''' make the wrap-around wiggle plot '''

    bins_per_wrap = int(periods_per_wrap *
                        approx_oma_period / hist.GetBinWidth(1))

    # build the hist segments and function segments
    hists = []
    funcs = []
    for start_bin in range(1, hist.GetNbinsX() - bins_per_wrap, bins_per_wrap):
        start_t = hist.GetBinLowEdge(start_bin)

        wrap_hist = r.TH1D(f'{fit.GetName()}wrap_start{start_bin}',
                           'wrapped_hist', bins_per_wrap,
                           hist.GetBinLowEdge(1),
                           hist.GetBinLowEdge(bins_per_wrap + 1))
        for i_bin in range(start_bin, start_bin + bins_per_wrap):
            # cut out bins before earliest time
            if hist.GetBinCenter(i_bin) > earliest_time:
                wrap_hist.SetBinContent(i_bin - start_bin + 1,
                                        hist.GetBinContent(i_bin))
                wrap_hist.SetBinError(i_bin - start_bin + 1,
                                      hist.GetBinError(i_bin))

        wrap_hist.SetMarkerSize(0.3)

        wrap_func_seg = r.TF1(
            f'{hist.GetName()}shifted{start_bin}',
            make_shifted_wiggle_func(fit),
            fit_start - start_t, fit_end - start_t, 1)
        wrap_func_seg.SetLineColor(r.kGreen + 2)
        wrap_func_seg.SetParameter(0, -start_t)
        wrap_func_seg.SetNpx(10000)

        hists.append(wrap_hist)
        funcs.append(wrap_func_seg)

    c = r.TCanvas()
    c.SetLogy(1)

    hists[0].Draw()
    funcs[0].Draw('same')
    t_per_wrap = hist.GetBinWidth(1) * bins_per_wrap
    hists[0].SetTitle(f'{title};time modulo {t_per_wrap:.0f}#mus;')
    for hist, func in zip(hists[1:], funcs[1:]):
        hist.Draw('same')
        func.Draw('same')

    y_min, y_max = 10, 1.5 * hists[0].GetMaximum()
    hists[0].GetYaxis().SetRangeUser(y_min, y_max)

    # try to put this text box in the upper right without
    # drawing it on top of the data
    txt = r.TPaveText(0.5 * t_per_wrap,
                      y_max * 0.8,
                      0.9 * t_per_wrap,
                      2 * hists[0].Interpolate(t_per_wrap * 0.5))
    txt.SetLineColor(r.kWhite)
    txt.SetShadowColor(r.kWhite)
    txt.SetFillColor(r.kWhite)

    txt.AddText(f'#chi^{{2}}/ndf: {fit.GetChisquare():.0f}/{fit.GetNDF():.0f}')
    txt.AddText(f'precision: {fit.GetParError(4):.2f} ppm')
    txt.Draw()

    return c, (hists, funcs, txt)
