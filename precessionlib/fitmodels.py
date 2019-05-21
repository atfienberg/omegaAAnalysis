# python versions of omega_a fit functions
#
# (and also some related/assocaited things)
#
# Aaron Fienberg
# September 2018

import ROOT as r
from .util import *
import os

# load some root functions and variables into the interpreter
file_dir = os.path.dirname(os.path.realpath(__file__))
r.gROOT.ProcessLine(f'.L {file_dir}/fitFunctions.C+')

# some strings for  building root TF1's
five_param_str = '[0]*exp(-x/[1])*(1+' +\
    '[2]*cos(x*[5]*(1+[4]*1e-6))+[3]*sin(x*[5]*(1+[4]*1e-6)))'

cbo_N_str = '(1+exp(-x/[6])*([7]*cos([9]*x) + [8]*sin([9]*x)))'
fit_with_cbo_str = f'{five_param_str}*{cbo_N_str}'

vw_N_str = '(1+exp(-x/[10])*([11]*cos([13]*x) + [12]*sin([13]*x)))'
fit_with_vw_str = f'{fit_with_cbo_str}*{vw_N_str}'

fit_with_losses_str = f'{fit_with_vw_str}*(1-[14]*cumuLoss(x))'


def build_5_param_func(config):
    five_param_tf1 = r.TF1('five_param_tf1', five_param_str, 0, 700)
    five_param_tf1.SetParName(0, 'N_{0}')
    five_param_tf1.SetParName(1, '#tau')
    five_param_tf1.SetParName(2, 'A_{C}')
    five_param_tf1.SetParName(3, 'A_{S}')
    five_param_tf1.SetParName(4, 'R')

    five_param_tf1.SetNpx(10000)
    five_param_tf1.SetLineColor(r.kRed)

    apply_fit_conf(five_param_tf1, config)

    return five_param_tf1


def build_CBO_only_func(five_par_f, cbo_freq, config):
    ''' builds a five params + CBO params TF1
    based on a five parameter fit function '''
    with_cbo_tf1 = r.TF1('with_cbo_TF1', fit_with_cbo_str, 0, 700)

    copy_all_parameters(five_par_f, with_cbo_tf1)

    # tau_cbo
    with_cbo_tf1.SetParName(6, '#tau_{CBO}')
    with_cbo_tf1.SetParameter(6, 150)

    # A_c,cbo
    with_cbo_tf1.SetParName(7, 'A_{CBO, c}')
    with_cbo_tf1.SetParameter(7, 0.004)

    # A_s,cbo
    with_cbo_tf1.SetParName(8, 'A_{CBO, s}')
    with_cbo_tf1.SetParameter(8, 0.004)

    # omega cbo
    with_cbo_tf1.SetParName(9, '#omega_{CBO}')
    with_cbo_tf1.SetParameter(9, cbo_freq * 2 * math.pi)

    with_cbo_tf1.SetNpx(10000)
    with_cbo_tf1.SetLineColor(r.kRed)

    apply_fit_conf(with_cbo_tf1, config)

    return with_cbo_tf1


def build_CBO_VW_func(cbo_f, cbo_freq, config, f_c=1.0 / 0.149):
    ''' builds a five params + CBO params + VW TF1
    based on a five params + CBO fit function '''
    with_vw_tf1 = r.TF1('with_vw_tf1', fit_with_vw_str, 0, 700)

    copy_all_parameters(cbo_f, with_vw_tf1)

    vw_freq = f_c * (1 - 2 * math.sqrt(n_of_CBO_freq(cbo_freq)))
    omega_vw = 2 * math.pi * vw_freq

    tau_cbo = cbo_f.GetParameter(6)

    # tau_vw, initiaize based on cbo lifetime
    with_vw_tf1.SetParName(10, '#tau_{vw}')
    with_vw_tf1.SetParameter(10, cbo_freq / vw_freq * tau_cbo)

    # A_c,vw
    with_vw_tf1.SetParName(11, 'A_{c, vw}')
    with_vw_tf1.SetParameter(11, 0.002)

    # A_s,vw
    with_vw_tf1.SetParName(12, 'A_{s, vw}')
    with_vw_tf1.SetParameter(12, 0.002)

    # omega vw
    with_vw_tf1.SetParName(13, '#omega_{vw}')
    with_vw_tf1.SetParameter(13, omega_vw)

    with_vw_tf1.SetNpx(10000)
    with_vw_tf1.SetLineColor(r.kRed)

    apply_fit_conf(with_vw_tf1, config)

    return with_vw_tf1


def loss_hist_is_initialized():
    return r.lossHistIsInitialized()


def build_losses_func(vw_f, config):
    ''' builds a function including VW, CBO, and muon losses
    based on a function including VW and CBO'''

    if not loss_hist_is_initialized():
        raise RuntimeError('Muon loss histogram is not initialized!')

    with_losses_tf1 = r.TF1('with_losses_tf1', fit_with_losses_str, 0, 700)

    copy_all_parameters(vw_f, with_losses_tf1)

    with_losses_tf1.SetParameter(14, 0)
    with_losses_tf1.SetParName(14, 'K_{loss}')

    with_losses_tf1.SetNpx(10000)
    with_losses_tf1.SetLineColor(r.kRed)

    apply_fit_conf(with_losses_tf1, config)

    return with_losses_tf1


def build_full_fit_tf1(loss_f, config, name='fullFit', f_c=1.0 / 0.149):
    '''build a full fit TF1, including CBO modulation of N, phi
    and changing CBO frequency
    this is implemented as a ROOT function in fitFunctions.C'''

    if not loss_hist_is_initialized():
        raise RuntimeError('Muon loss histogram is not initialized!')

    r.createFullFitTF1(name)
    full_fit_tf1 = r.gROOT.FindObject(name)

    copy_all_parameters(loss_f, full_fit_tf1)

    # rename and reset parameters
    # that are now using A, phi parameterization
    AcAs_to_APhi(full_fit_tf1, 2, 3)
    full_fit_tf1.SetParName(2, 'A')
    full_fit_tf1.SetParName(3, '#phi')

    AcAs_to_APhi(full_fit_tf1, 7, 8)
    full_fit_tf1.SetParName(7, 'A_{CBO}')
    full_fit_tf1.SetParName(8, '#phi_{CBO}')

    AcAs_to_APhi(full_fit_tf1, 11, 12)
    full_fit_tf1.SetParName(11, 'A_{vw}')
    full_fit_tf1.SetParName(12, '#phi_{vw}')

    # cbo modulation on A and phi
    full_fit_tf1.SetParName(15, 'A_{CBO, A}')
    full_fit_tf1.SetParName(16, '#phi_{CBO, A}')
    full_fit_tf1.SetParName(17, 'A_{CBO, #phi}')
    full_fit_tf1.SetParName(18, '#phi_{CBO, #phi}')
    full_fit_tf1.SetParName(19, 'd#omega_{CBO}/dt')

    for par_num in range(15, 20):
        full_fit_tf1.SetParameter(par_num, 0)

    # frequency slope
    freq_params = config['cbo_freq_params']

    full_fit_tf1.SetParName(31, 'Tracker Model Number')
    full_fit_tf1.FixParameter(31, freq_params['model_num'])

    if freq_params['fix_slope']:
        full_fit_tf1.FixParameter(19, freq_params['m'])
    else:
        full_fit_tf1.SetParameter(19, freq_params['m'])

    try:
        full_fit_tf1.SetParameter(9, freq_params['freq_guess'])
    except KeyError:
        pass

    # parameters 20-23 come from the trackers
    full_fit_tf1.FixParameter(20, freq_params['A'])
    full_fit_tf1.FixParameter(21, freq_params['B'])
    full_fit_tf1.FixParameter(22, freq_params['tau_a'])
    full_fit_tf1.FixParameter(23, freq_params['tau_b'])

    # parameters 24-26 are the 2 omega_cbo parameters
    # can be important for fitting single calorimeters

    # start by fixing them, but we can release them
    # for the single calorimeter fits
    full_fit_tf1.SetParName(24, '#tau_{2CBO}')
    full_fit_tf1.FixParameter(24, full_fit_tf1.GetParameter(6) / 2)
    full_fit_tf1.SetParName(25, 'A_{2CBO}')
    full_fit_tf1.FixParameter(25, 0)
    full_fit_tf1.SetParName(26, '#phi_{2CBO}')
    full_fit_tf1.FixParameter(26, 0)

    # now set vertical betatron parameters.
    # These default to being effectively excluded from the fit
    include_y_osc = config.get('include_y_osc', False)

    # vertical betatron oscillations
    full_fit_tf1.SetParName(27, '#tau_{y}')
    full_fit_tf1.SetParName(28, 'A_{y}')
    full_fit_tf1.SetParName(29, '#phi_{y}')
    full_fit_tf1.SetParName(30, '#omega_{y}')

    if include_y_osc:
        cbo_freq = full_fit_tf1.GetParameter(9) / 2 / math.pi
        y_freq = f_c * math.sqrt(n_of_CBO_freq(cbo_freq))
        w_y = 2 * math.pi * y_freq

        full_fit_tf1.SetParameter(
            27, cbo_freq / y_freq * full_fit_tf1.GetParameter(6))
        full_fit_tf1.SetParameter(28, 0.0001)
        full_fit_tf1.SetParameter(29, 0)
        full_fit_tf1.SetParameter(30, w_y)

    else:
        for par_num in range(27, 31):
            full_fit_tf1.FixParameter(par_num, 0)

    full_fit_tf1.SetNpx(10000)
    full_fit_tf1.SetLineColor(r.kRed)

    # whether to switch to "use field index" mode
    full_fit_tf1.SetParName(32, 'Use Field Index Mode')
    use_field_index = config.get('field_index_mode', False)
    full_fit_tf1.FixParameter(32, 1.0 if use_field_index else 0.0)

    if use_field_index:
        # switch to field index mode
        full_fit_tf1.SetParName(13, '#delta_{vw}')
        full_fit_tf1.SetParameter(13, 0)

        full_fit_tf1.SetParName(30, '#delta_{y}')
        if include_y_osc:
            full_fit_tf1.SetParameter(13, 0)

    for [par_name, val] in config.get('fit_par_guesses', []):
        par_num = get_par_index(full_fit_tf1, par_name)
        if not is_free_param(full_fit_tf1, par_num):
            full_fit_tf1.ReleaseParameter(par_num)
        full_fit_tf1.SetParameter(par_num, val)

    pars_to_limit = config.get('fit_par_limits', [])
    for [par_name, low, high] in pars_to_limit:
        par_num = get_par_index(full_fit_tf1, par_name)
        full_fit_tf1.SetParLimits(par_num, low, high)

    pars_to_fix = config.get('fit_par_fixes', [])
    for [par_name, val] in pars_to_fix:
        par_num = get_par_index(full_fit_tf1, par_name)
        full_fit_tf1.FixParameter(par_num, val)

    return full_fit_tf1


def clone_full_fit_tf1(full_fit, name):
    ''' clone a full fit tf1 and return it
        TF1::Clone() was not working for me
    '''

    if not loss_hist_is_initialized():
        raise RuntimeError('Muon loss histogram is not initialized!')

    r.createFullFitTF1(name)
    new_fit = r.gROOT.FindObject(name)

    copy_all_parameters(full_fit, new_fit)

    new_fit.SetNpx(10000)
    new_fit.SetLineColor(r.kRed)

    return new_fit


def apply_fit_conf(func, config):
    '''apply full fit conf to a function that is not the full_fit_tf1,
    e.g. the CBO TF1 or vw TF1'''
    for [par_name, val] in config.get('fit_par_guesses', []):
        try:
            par_num = get_par_index(func, par_name)
        except ValueError:
            continue

        func.SetParameter(par_num, val)

    pars_to_limit = config.get('fit_par_limits', [])
    for [par_name, low, high] in pars_to_limit:
        try:
            par_num = get_par_index(func, par_name)
        except ValueError:
            continue

        func.SetParLimits(par_num, low, high)

    pars_to_fix = config.get('fit_par_fixes', [])
    for [par_name, val] in pars_to_fix:
        try:
            par_num = get_par_index(func, par_name)
        except ValueError:
            continue

        func.FixParameter(par_num, val)


def prepare_loss_hist(config, T_meth_hist, tau=64.44):
    '''prepare the muon loss histogram needed for fitting
    also make some plots'''
    lost_muon_rate_2d = get_histogram(
        config['loss_hist_name'], config['loss_file_name'])
    lost_muon_rate_2d.SetTitle(
        'triple coincidence counts; time [#mu s]; N [a.u.]')

    # # for guaranteeing that nothing matters before t=t_s
    # for x_bin in range(1, lost_muon_rate_2d.GetNbinsX() + 1):
    #     if lost_muon_rate_2d.GetXaxis().GetBinCenter(x_bin) > 27:
    #         continue
    #     for y_bin in range(1, lost_muon_rate_2d.GetNbinsY() + 1):
    #         lost_muon_rate_2d.SetBinContent(x_bin, y_bin, 0)

    # create lost muon histogram with bin contents weighted by exp(t/tau)
    lost_muon_prob_2d = lost_muon_rate_2d.Clone()
    lost_muon_prob_2d.SetName('lost_muon_prob')
    for x_bin in range(1, lost_muon_rate_2d.GetNbinsX()):
        time_center = lost_muon_prob_2d.GetXaxis().GetBinCenter(x_bin)

        for y_bin in range(1, lost_muon_rate_2d.GetNbinsY()):

            new_content = math.exp(time_center / tau) * \
                lost_muon_prob_2d.GetBinContent(x_bin, y_bin)
            lost_muon_prob_2d.SetBinContent(x_bin, y_bin, new_content)

    lost_muon_rate = lost_muon_rate_2d.ProjectionX()
    lost_muon_prob = lost_muon_prob_2d.ProjectionX()
    # clone the prob hist before scaling for plot
    # so that we have a meaningful scaling in the fits
    unscaled_prob = lost_muon_prob.Clone()
    unscaled_prob.SetName('unscaled_loss_prob')

    # now build a meaninfully scaled cumulative loss hist to use for fitting
    lost_muon_cumulative = unscaled_prob.GetCumulative()
    lost_muon_cumulative.SetName('cumulativeLossHist')
    # should scale by inverse of estimated number of positrons in the dataset
    # that is about 10 times the number in the T-method hist,
    # multipled by 1.6 to extrapolate from 30 microseconds to t = 0
    lost_muon_cumulative.Scale(
        1.0 / (after_t(T_meth_hist, 30).Integral() * 16))

    # initialize the pointer used in the cumuLoss function
    # initializeLossHist is defined in fitFunctions.C
    r.initializeLossHist(lost_muon_cumulative.GetName())

    # return histograms
    return lost_muon_cumulative, lost_muon_rate, lost_muon_prob
