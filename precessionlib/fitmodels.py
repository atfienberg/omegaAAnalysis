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


def build_5_param_func():
    five_param_tf1 = r.TF1('five_param_tf1', five_param_str, 0, 700)
    five_param_tf1.SetParName(0, 'N_{0}')
    five_param_tf1.SetParName(1, '#tau')
    five_param_tf1.SetParName(2, 'A_{C}')
    five_param_tf1.SetParName(3, 'A_{S}')
    five_param_tf1.SetParName(4, 'R')

    five_param_tf1.SetNpx(10000)
    five_param_tf1.SetLineColor(r.kRed)

    return five_param_tf1


def build_CBO_only_func(five_par_f, cbo_freq):
    ''' builds a five params + CBO params TF1
    based on a five parameter fit function '''
    with_cbo_tf1 = r.TF1('with_cbo_TF1', fit_with_cbo_str, 0, 700)
    for par_num in range(five_par_f.GetNpar()):
        with_cbo_tf1.SetParameter(
            par_num, five_par_f.GetParameter(par_num))
        with_cbo_tf1.SetParName(par_num, five_par_f.GetParName(par_num))

    with_cbo_tf1.FixParameter(5, with_cbo_tf1.GetParameter(5))

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

    return with_cbo_tf1


def build_CBO_VW_func(cbo_f, cbo_freq, f_c=1.0 / 0.149):
    ''' builds a five params + CBO params + VW TF1
    based on a five params + CBO fit function '''
    with_vw_tf1 = r.TF1('with_vw_tf1', fit_with_vw_str, 0, 700)
    for par_num in range(cbo_f.GetNpar()):
        with_vw_tf1.SetParameter(par_num, cbo_f.GetParameter(par_num))
        with_vw_tf1.SetParName(par_num, cbo_f.GetParName(par_num))

    with_vw_tf1.FixParameter(5, with_vw_tf1.GetParameter(5))

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

    return with_vw_tf1


def loss_hist_is_initialized():
    return r.lossHistIsInitialized()


def build_losses_func(vw_f):
    ''' builds a function including VW, CBO, and muon losses
    based on a function including VW and CBO'''

    if not loss_hist_is_initialized():
        raise RuntimeError('Muon loss histogram is not initialized!')

    with_losses_tf1 = r.TF1('with_losses_tf1', fit_with_losses_str, 0, 700)

    for par_num in range(vw_f.GetNpar()):
        with_losses_tf1.SetParameter(par_num, vw_f.GetParameter(par_num))
        with_losses_tf1.SetParName(par_num, vw_f.GetParName(par_num))

    with_losses_tf1.FixParameter(5, with_losses_tf1.GetParameter(5))

    with_losses_tf1.SetParameter(14, 0)
    with_losses_tf1.SetParName(14, 'K_{loss}')

    with_losses_tf1.SetNpx(10000)
    with_losses_tf1.SetLineColor(r.kRed)

    return with_losses_tf1


def build_full_fit_tf1(loss_f, config, name='fullFit'):
    # build a full fit TF1, including CBO modulation of N, phi
    # and changing CBO frequency
    # this is implemented as a ROOT function in fitFunctions.C

    if not loss_hist_is_initialized():
        raise RuntimeError('Muon loss histogram is not initialized!')

    r.createFullFitTF1(name)
    full_fit_tf1 = r.gROOT.FindObject(name)

    for par_num in range(loss_f.GetNpar()):
        full_fit_tf1.SetParameter(par_num, loss_f.GetParameter(par_num))
        full_fit_tf1.SetParName(par_num, loss_f.GetParName(par_num))

    full_fit_tf1.FixParameter(5, full_fit_tf1.GetParameter(5))

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

    if freq_params['fix_slope']:
        full_fit_tf1.FixParameter(19, freq_params['m'])
    else:
        full_fit_tf1.SetParameter(19, freq_params['m'])

    # parameters 20-23 come from the trackers
    # they should be in the config, but I'll hard code them
    # for this initial test
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

    full_fit_tf1.SetNpx(10000)
    full_fit_tf1.SetLineColor(r.kRed)

    return full_fit_tf1


def clone_full_fit_tf1(full_fit, name):
    ''' clone a full fit tf1 and return it
        TF1::Clone() was not working for me
    '''

    if not loss_hist_is_initialized():
        raise RuntimeError('Muon loss histogram is not initialized!')

    r.createFullFitTF1(name)
    new_fit = r.gROOT.FindObject(name)

    for par_num in range(new_fit.GetNpar()):
        new_fit.SetParName(par_num, full_fit.GetParName(par_num))

        par_val = full_fit.GetParameter(par_num)

        if is_free_param(full_fit, par_num):
            new_fit.SetParameter(par_num, par_val)

            low, high = r.Double(), r.Double()
            full_fit.GetParLimits(par_num, low, high)
            if not (low == 0 and high == 0):
                new_fit.SetParLimits(par_num, low, high)

        else:
            new_fit.FixParameter(par_num, par_val)

    new_fit.SetNpx(10000)
    new_fit.SetLineColor(r.kRed)

    return new_fit


def prepare_loss_hist(config, T_meth_hist, tau=64.44):
    '''prepare the muon loss histogram needed for fitting
    also make some plots'''
    lost_muon_rate_2d = get_histogram(
        config['loss_hist_name'], config['loss_file_name'])
    lost_muon_rate_2d.SetTitle(
        'triple coincidence counts; time [#mu s]; N [a.u.]')

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
