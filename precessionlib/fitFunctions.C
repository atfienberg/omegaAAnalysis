// functions needed for fitting the precession histograms
//
// Aaron Fienberg
// September 2018

#include "TH1.h"
#include "TF1.h"

// the histogram used for including the muon loss term
TH1D* cumuLossHist = nullptr;

bool lossHistIsInitialized() { return cumuLossHist != nullptr; }

void initializeLossHist(const char* histName) {
  cumuLossHist = (TH1D*)gROOT->FindObject(histName);
}

// do not call this before initializing the loss hist ptr
double cumuLoss(double x) { return cumuLossHist->Interpolate(x); }

// constexpr unsigned int n_full_fit_parameters = 27;
constexpr unsigned int n_full_fit_parameters = 27;
double full_wiggle_fit(double* x, double* p) {
  double t = x[0];
  double N_0 = p[0];
  double tau = p[1];
  double A = p[2];
  double phi = p[3];
  double R = p[4];
  double wa_ref = p[5];
  double tau_cbo = p[6];
  double A_cbo = p[7];
  double phi_cbo = p[8];
  double w_cbo = p[9];
  double tau_vw = p[10];
  double A_vw = p[11];
  double phi_vw = p[12];
  double w_vw = p[13];
  double K_loss = p[14];

  double A_cboa = p[15];
  double phi_cboa = p[16];
  double A_cbophi = p[17];
  double phi_cbophi = p[18];

  double N_loss = 1 - K_loss * cumuLoss(t);
  double N_vw = 1 + exp(-t / tau_vw) * (A_vw * cos(w_vw * t - phi_vw));

  // tracker cbo frequency changes
  // parameters [19-23] come from the tracker
  // they are the parameters for modeling the
  // changing CBO frequency
  double dw = p[19] / 100 / 1000;
  double cbo_exp_a = p[20] / 100;
  double cbo_exp_b = p[21] / 100;
  double tau_a = p[22];
  double tau_b = p[23];

  w_cbo = w_cbo * (1 + dw * t + cbo_exp_a * exp(-t / tau_a) +
                   cbo_exp_b * exp(-t / tau_b));

  // test using double omega_CBO parameters on the n term
  double tau_2cbo = p[24];
  double A_2cbo = p[25];
  double phi_2cbo = p[26];

  // assymetry/phase modulation
  A = A * (1 + exp(-t / tau_cbo) * A_cboa * cos(w_cbo * t - phi_cboa));
  phi = phi + exp(-t / tau_cbo) * A_cbophi * cos(w_cbo * t - phi_cbophi);

  double N_cbo = 1 + exp(-t / tau_cbo) * (A_cbo * cos(w_cbo * t - phi_cbo));

  double N_2cbo =
      1 + exp(-t / tau_2cbo) * (A_2cbo * cos(2 * w_cbo * t - phi_2cbo));

  double N = N_0 * N_cbo * N_2cbo * N_loss * N_vw;

  double wa = wa_ref * (1 + R * 1e-6);

  return N * exp(-t / tau) * (1 + A * cos(wa * t - phi));
}

void createFullFitTF1(const char* tf1Name) {
  new TF1(tf1Name, full_wiggle_fit, 0, 700, n_full_fit_parameters);
}