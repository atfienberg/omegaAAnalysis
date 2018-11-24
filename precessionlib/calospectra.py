# provides the CaloSpectra class
#
# Aaron Fienberg
# September 2018


import sys
import ctypes
import ROOT as r
import numpy as np
import subprocess
import os
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm


class CaloSpectra:
    ''' stores energy versus time spectrum for all calos as a 3d numpy array
    also stores edges for x, y, z axes

    build using either from_root_file or from_np_file
    default constructor will give you nothing useful

    constructors take parameters used for building pileup spectra:
    do_triple: whether to include the triple pileup contribution
    pu_energy_min/max, pu_time_min: normalization parameters.
    The pileup distribution is normalized such that the integrals
    of the pileup correction and uncorrected spectrum over configured
    time and energy ranges are equal.
    '''

    def __init__(self,
                 do_triple=False,
                 pu_energy_min=3500, pu_energy_max=6000,
                 pu_time_min=30, single_param=False, param_guess=None):
        self._do_triple = do_triple
        self._pu_emin = pu_energy_min
        self._pu_emax = pu_energy_max
        self._pu_tmin = pu_time_min

        self.single_param = single_param
        if single_param:
            if param_guess is None:
                raise ValueError(
                    'must provide param_guess when using single_param=True')
            self.param_guess = param_guess

    @staticmethod
    def from_root_file(rootfilename, histname='master3D',
                       do_triple=False,
                       pu_energy_min=4000, pu_energy_max=6000,
                       pu_time_min=30, single_param=False, param_guess=None):
        ''' load 3d histogram from rootfile
        store numpy version of 3d hist + x, y, z axes'''
        out = CaloSpectra(do_triple, pu_energy_min, pu_energy_max,
                          pu_time_min, single_param, param_guess)

        file = r.TFile(rootfilename)
        hist = file.Get(histname)

        cwd = r.gDirectory
        last_slash = histname.rfind('/')
        if last_slash != -1:
            histdir = file.Get(histname[:last_slash])
            histdir.cd()

        out._axes = [CaloSpectra.build_axis(hist, i) for i in range(3)]

        out._array = CaloSpectra.build_hist_array(hist)

        cwd.cd()

        return out

    @staticmethod
    def from_np_file(numpyfilename,
                     do_triple=False,
                     pu_energy_min=4000, pu_energy_max=6000,
                     pu_time_min=30, single_param=False, param_guess=None):
        out = CaloSpectra(do_triple, pu_energy_min, pu_energy_max, pu_time_min)

        loaded = np.load(numpyfilename)
        out._axes = [loaded['calo'], loaded['energy'], loaded['time']]
        out._array = loaded['array']

        return out

    def save_to_file(self, filename):
        with open(filename, 'wb') as f:
            np.savez_compressed(f, array=self.array,
                                calo=self.axes[0],
                                energy=self.axes[1],
                                time=self.axes[2])

    @property
    def calo_sum(self):
        try:
            return self._calo_sum
        except AttributeError:
            self._calo_sum = self.array.sum(axis=0)
            return self._calo_sum

    @property
    def pu_spectrum(self):
        try:
            return self._pu_array
        except AttributeError:
            self._build_pu_array()
            return self._pu_array

    @property
    def pu_coeffs(self):
        '''get pileup scale coefficients
        gives a list, each element is for one calo'''
        try:
            return self._pu_coeffs
        except AttributeError:
            self._build_pu_array()
            return self._pu_coeffs

    @property
    def pu_covs(self):
        '''get pileup scale factor covariance matrices
        gives a list, each element is for one calo'''
        try:
            return self._pu_covs
        except AttributeError:
            self._build_pu_array()
            return self._pu_covs

    @property
    def cor_variances(self):
        '''returns variance for each bin in the corrected histogram
        corrected histogram is self.array - self.pu_spectrum'''
        try:
            return self._cor_variances
        except AttributeError:
            self._build_cor_variances()
            return self._cor_variances

    @property
    def var_cor_factors(self):
        '''returns variance correction factors'''
        try:
            return self._var_cor_factors
        except AttributeError:
            self._build_cor_variances()
            return self._var_cor_factors

    @property
    def time_centers(self):
        try:
            return self._time_centers
        except AttributeError:
            self._time_centers = 0.5 * (self.axes[2][:-1] + self.axes[2][1:])
            return self._time_centers

    @property
    def energy_centers(self):
        try:
            return self._energy_centers
        except AttributeError:
            self._energy_centers = 0.5 * (self.axes[1][:-1] + self.axes[1][1:])
            return self._energy_centers

    def __getitem__(self, index):
        '''get an individual calo spectrum by index,
        self[0] is the spectrum for calo 1'''
        if index >= self.array.shape[0]:
            raise IndexError
        return self.array[index, :, :]

    def calo_spec(self, calo_num):
        ''' get an individual calo spectrum by calo number'''
        return self[calo_num - 1]

    def build_delta_rho_pu(self, calo_num, rho_guess=None, do_triple=None,
                           ret_comps=False,
                           pu_energy_min=None, pu_energy_max=None,
                           pu_time_min=None):
        '''build the delta_rho_pileup distribution for calo calo_num.
        if rho_guess is None, rho will be calo_spec(calo_num)
        otherwise, rho_guess will be the positron distribution.
        This allows for multiple iterations of the corretion procedure, e.g.:
        drho_pu_1iter = spec.build_delta_rho_pu(calo_num=1)
        drho_pu_2iters = spec.build_delta_rho_pu(
            rho_guess=spec.calo_spec(1)-drho_pu_1iter)

        if ret_comps is true, returns (delta_rho_pu, norm_factors, cov, comps)
        norm_factors are the normalization factors for the pileup components
        cov the norm_factors covariance matrix
        comps are the pu spectrum components, e.g. [double pu, triple...]

        if ret_comps is false, returns just delta_rho_pu, norm_factors, cov

        pu_energy_min/max, pu_time_min determine range for fitting pu spectrum
        The pileup distribution is fit to the spectrum in the range
        t > pu_time min, pu_energy_min < E < pileup_energy_max
        so, pu_energy_min should be above the expected energy spectrum endpoint

        normalization params default to those specified in constructor
        '''

        if do_triple is None:
            do_triple = self._do_triple
        if pu_energy_min is None:
            pu_energy_min = self._pu_emin
        if pu_energy_max is None:
            pu_energy_max = self._pu_emax
        if pu_time_min is None:
            pu_time_min = self._pu_tmin

        # rho is the hit distribution from which we build the pu distribution
        rho = self.calo_spec(calo_num) if rho_guess is None else rho_guess

        drho_pu_plus, drho_pu_minus = CaloSpectra._build_double_pu_specs(rho)
        drho_pu_double = drho_pu_plus + drho_pu_minus

        components = [drho_pu_double]

        if do_triple:
            trip_comps = CaloSpectra._build_triple_pu_specs(rho, drho_pu_plus)
            components.append(sum(trip_comps))

        # get normalization coefficients
        try:
            norm_factors, cov = self._fit_pu_coeffs(
                calo_num, components,
                pu_energy_min, pu_energy_max, pu_time_min)
        except ValueError:
            # no high count pileup bins, set the coefficients to zero
            print('Failure to fit pileup coefficients'
                  f' for z bin {calo_num}: setting to 0')
            norm_factors = np.zeros(len(components))
            cov = np.identity(len(components))

        # apply the normalization factors
        for comp, norm in zip(components, norm_factors):
            comp *= norm

        if ret_comps:
            return sum(components), norm_factors, cov, components
        else:
            return sum(components), norm_factors, cov

    @staticmethod
    def _build_double_pu_specs(rho):
        '''builds the non-normalized double pileup contributions
        returns: pos_component, neg_component
        '''

        # plus term:
        # sum over rho(t, E2) * rho(t, E-E2)
        # bin i for E2, bin e_bin - 1 - i  for bin E - E2
        #
        # bins where e_bin - 1 - i <= 0 are not included,
        # those have E - E2 <= 0
        drho_pu_plus = np.empty_like(rho)
        for e_bin in range(rho.shape[0]):
            drho_pu_plus[e_bin] = np.einsum(
                'ij,ij -> j', rho[:e_bin, :], rho[:e_bin, :][::-1])

        # minus term:
        # sum over rows of rho(t, E) * rho(t, E2)
        # has a factor of two
        drho_pu_minus = -2 * rho * rho.sum(axis=0)

        return drho_pu_plus, drho_pu_minus

    @staticmethod
    def _build_triple_pu_specs(rho, double_plus):
        '''builds the non-normalized triple pileup contributions
        returns: tripleSum, doubleCorrection, singleCorrection
        '''
        rho_ints = rho.sum(axis=0)

        # distribution of sum of three positron energies
        trip_sum = np.empty_like(rho)
        for e_bin in range(rho.shape[0]):
            trip_sum[e_bin] = np.einsum(
                'ij,ij -> j', double_plus[:e_bin, :], rho[:e_bin, :][::-1])

        # correct for positive double pileup that was part of a triple
        trip_term_two = -3 * double_plus * rho_ints

        # correct for negative double pileup that was part of a triple
        trip_term_three = 3 * rho * np.power(rho_ints, 2)

        return trip_sum, trip_term_two, trip_term_three

    def _fit_pu_coeffs(self, calo_num, pu_comps,
                       pu_energy_min, pu_energy_max, pu_time_min,
                       min_counts=20):
        ''' fit the scale/normalization factors for the pileup components
        pu_comps is a list of the pileup components, e.g. [doubles, triples]
        returns: (coefficients list, covariance matrix)
        coefficients list is the same length as the pu_comps'''
        energies = np.logical_and(
            self.energy_centers > pu_energy_min,
            self.energy_centers < pu_energy_max)
        times = self.time_centers > pu_time_min

        # perturbed energy spectrum
        perturbed = self.calo_spec(calo_num)[:, times][energies, :].sum(axis=1)

        # to keep fit reasonable,
        # do not include bins with fewer than min_counts counts in the fit
        high_count_bins = perturbed > min_counts
        perturbed = perturbed[high_count_bins]

        # component energy spectra
        components = np.hstack(comp[:, times][energies, :].sum(axis=1)[
            :, None] for comp in pu_comps)
        # remove the bins that had low counts
        components = components[high_count_bins, :]

        if self.single_param:
            # if fitting with a single parameter, use curve_fit
            # because the fit is not linear
            def fit_f(x, eps):
                retval = np.zeros_like(x, dtype='float64')
                for i, _ in enumerate(pu_comps, 1):
                    retval += eps**i * components.T[i - 1]

                return retval

            comp, cov = curve_fit(
                fit_f, np.arange(perturbed.shape[0]),
                perturbed, p0=self.param_guess, sigma=np.sqrt(perturbed))

            coeffs = np.array([comp[0]**i for i, _ in enumerate(pu_comps, 1)])
            covariance = cov

        else:
            # if fitting with linear combo of comps,
            # we can use linear least squares

            # use Poisson errors,
            # sqrt of the bin content from the perturbed energy spec
            errors = np.sqrt(perturbed)
            components *= 1 / errors[:, None]
            perturbed *= 1 / errors

            coeffs, _, _, _ = np.linalg.lstsq(components, perturbed)

            covariance = np.linalg.inv(components.T @ components)

        return coeffs, covariance

    def plot_all_calos(self):
        plt.pcolormesh(self.time_centers, self.energy_centers, self.calo_sum,
                       norm=LogNorm(vmin=1, vmax=self.calo_sum.max()),
                       cmap='seismic')
        plt.xlabel('time [$\mu$ s]', fontsize=24)
        plt.ylabel('energy [MeV]', fontsize=24)
        plt.title('all calos', fontsize=24)
        plt.tick_params(labelsize=24)
        plt.colorbar()

    def plot_one_calo(self, calo_num):
        plt.pcolormesh(self.time_centers, self.energy_centers,
                       self.calo_spec(calo_num),
                       norm=LogNorm(vmin=1, vmax=self.calo_sum.max()),
                       cmap='seismic')
        plt.xlabel('time [$\mu$ s]', fontsize=24)
        plt.ylabel('energy [MeV]', fontsize=24)
        plt.title('calo {}'.format(calo_num), fontsize=24)
        plt.tick_params(labelsize=24)
        plt.colorbar()

    @property
    def axes(self):
        return self._axes

    @property
    def array(self):
        return self._array

    @staticmethod
    def build_axis(hist3d, axis):
        '''build and return numpy array of axis edges'''
        if axis == 2:
            return CaloSpectra.build_x_axis(hist3d)
        elif axis == 1:
            return CaloSpectra.build_y_axis(hist3d)
        elif axis == 0:
            return CaloSpectra.build_z_axis(hist3d)
        else:
            raise ValueError('axis must be 0, 1, or 2!')

    @staticmethod
    def build_x_axis(hist3d):
        return CaloSpectra.build_axis_array(hist3d.GetXaxis())

    @staticmethod
    def build_y_axis(hist3d):
        return CaloSpectra.build_axis_array(hist3d.GetYaxis())

    @staticmethod
    def build_z_axis(hist3d):
        return CaloSpectra.build_axis_array(hist3d.GetZaxis())

    @staticmethod
    def build_axis_array(axis):
        array = np.empty(axis.GetNbins() + 1)
        for i_bin in range(1, axis.GetNbins() + 2):
            array[i_bin - 1] = axis.GetBinLowEdge(i_bin)

        return array

    # load C functions for filling a numpy array from a root 3d hist
    # and vice versa
    file_dir = os.path.dirname(os.path.realpath(__file__))
    try:
        _histConvertLib = ctypes.cdll.LoadLibrary(
            f'{file_dir}/hist3dToNumpyArray.so')
    except OSError:
        # try to compile the needed C++ code
        cwd = os.getcwd()
        os.chdir(file_dir)
        subprocess.call(['make'])
        _histConvertLib = ctypes.cdll.LoadLibrary(
            './hist3dToNumpyArray.so')
        os.chdir(cwd)

    _fill_array_from_hist = _histConvertLib.hist3dToNumpyArray
    _fill_array_from_hist.restype = ctypes.c_int
    _fill_array_from_hist.argtypes = [np.ctypeslib.ndpointer(
        ctypes.c_double, flags=('C_CONTIGUOUS', 'WRITEABLE')),
        ctypes.c_size_t, ctypes.c_char_p]

    _fill_hist_from_array = _histConvertLib.numpyArrayToHist3d
    _fill_hist_from_array.restype = ctypes.c_int
    _fill_hist_from_array.argtypes = [np.ctypeslib.ndpointer(
        ctypes.c_double, flags=('C_CONTIGUOUS', 'WRITEABLE')),
        ctypes.c_size_t, ctypes.c_char_p]

    _set_hist_errors = _histConvertLib.fillHist3dErrors
    _set_hist_errors.restype = ctypes.c_int
    _set_hist_errors.argtypes = [np.ctypeslib.ndpointer(
        ctypes.c_double, flags=('C_CONTIGUOUS', 'WRITEABLE')),
        ctypes.c_size_t, ctypes.c_char_p]

    @staticmethod
    def build_hist_array(hist3d):
        '''builds a 3d numpy hist from the passed-in root 3d histogram'''
        shape = (hist3d.GetNbinsZ(), hist3d.GetNbinsY(), hist3d.GetNbinsX())
        array = np.empty(shape)

        # this loop was super, super slow for big 3d hists
        # so, I outsourced it to a C function (loaded above)
        #
        # for (z,y,x) in itertools.product(*[range(1,n+1) for n in shape]):
        #     array[z - 1, y - 1, x - 1] = hist3d.GetBinContent(x, y, z)

        retcode = CaloSpectra._fill_array_from_hist(
            array, array.size, hist3d.GetName().encode())

        if retcode == -1:
            raise ValueError(
                'Can not find hist with name {}'.format(hist3d.GetName()))
        elif retcode == -2:
            raise ValueError('array size does not match 3d hist size!')

        return array

    @staticmethod
    def build_root_hist(array, bin_edges, histname):
        '''build a 3D root hist from the passed in array
        shape(array) must match that suggested by bin_edges
        bin_edges is a list of edges for the z, y, x axes (in that order)
        for a CaloSpectra called spec, a valid call to this function would be:
        build_root_hist(spec.array, spec.axes, 'newhist')

        Purpose here is to build a 3D root hist including pileup correction'''
        axis_limits = sum(([axis.size - 1, axis[0], axis[-1]]
                           for axis in bin_edges[::-1]), [])

        hist = r.TH3D(histname, histname, *axis_limits)

        retcode = CaloSpectra._fill_hist_from_array(
            array, array.size, histname.encode())

        if retcode == -1:
            raise ValueError(
                'Can not find hist with name {}'.format(histname))
        elif retcode == -2:
            raise ValueError('array size does not match 3d hist size!')

        return hist

    @staticmethod
    def set_hist_errors(error_array, hist3d):
        '''set bin 3d hist bin errors based on error_array'''
        retcode = CaloSpectra._set_hist_errors(error_array, error_array.size,
                                               hist3d.GetName().encode())

        if retcode == -1:
            raise ValueError(
                'Can not find hist with name {}'.format(hist3d.GetName()))
        elif retcode == -2:
            raise ValueError('error array size does not match 3d hist size!')

    def _build_pu_array(self):
        '''build pu spectrum for each calorimeter using configured params
           stores results and normalization factors in self

           triple pileup correction require 2 iterations
           double pileup correction requires only one
        '''
        self._pu_array = np.empty_like(self.array)
        n_calos = self.array.shape[0]
        self._pu_coeffs = [[] for i in range(n_calos)]
        self._pu_covs = [[] for i in range(n_calos)]

        n_iters = 1 if self._do_triple is False else 2

        for calo_num in range(1, n_calos + 1):
            pu_pert = 0
            for n in range(n_iters):
                guess = self.calo_spec(calo_num) - pu_pert
                pu_pert, norms, cov = self.build_delta_rho_pu(calo_num, guess)

            self._pu_array[calo_num - 1] = pu_pert
            self._pu_coeffs[calo_num - 1] = norms
            self._pu_covs[calo_num - 1] = cov

    def _build_cor_variances(self):
        ''' build variances for each bin in the pileup corrected spectrum
            neglects order triple pileup contribution to the bin uncertainties
        '''
        correction_factors = self.array.sum(axis=1)
        for i, norms in enumerate(self.pu_coeffs):
            correction_factors[i] = 1 + 4 * norms[0] * correction_factors[i]
        self._var_cor_factors = correction_factors

        self._cor_variances = np.array(self.array)
        for calo_vars, factor in zip(self._cor_variances, correction_factors):
            calo_vars *= factor


def main():
    spec = CaloSpectra.from_np_file('heavyNumpySpec.npz')

    spec.plot_all_calos()
    plt.show()


if __name__ == '__main__':
    sys.exit(main())
