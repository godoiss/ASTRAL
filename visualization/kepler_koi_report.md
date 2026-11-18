# Kepler KOI exploratory report

- Generated at (UTC): 2026-11-16T17:36:55.721626+00:00
- Source: https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+%2A+from+cumulative&format=csv
- Records: 9564
- Columns: 153
- Raw-data transformations: none

## Original KOI disposition distribution

| koi_disposition | count | percentage |
| --- | --- | --- |
| FALSE POSITIVE | 4839 | 50.596 |
| CONFIRMED | 2748 | 28.7327 |
| CANDIDATE | 1977 | 20.6713 |

## Missing values

Only columns with at least one missing value are listed.

| column | missing_count | missing_percentage |
| --- | --- | --- |
| koi_gmag_err | 9564 | 100 |
| koi_imag_err | 9564 | 100 |
| koi_zmag_err | 9564 | 100 |
| koi_rmag_err | 9564 | 100 |
| koi_kepmag_err | 9564 | 100 |
| koi_longp_err2 | 9564 | 100 |
| koi_sage_err2 | 9564 | 100 |
| koi_sage_err1 | 9564 | 100 |
| koi_sage | 9564 | 100 |
| koi_sma_err1 | 9564 | 100 |
| koi_sma_err2 | 9564 | 100 |
| koi_ingress_err1 | 9564 | 100 |
| koi_ingress | 9564 | 100 |
| koi_ingress_err2 | 9564 | 100 |
| koi_incl_err2 | 9564 | 100 |
| koi_teq_err1 | 9564 | 100 |
| koi_teq_err2 | 9564 | 100 |
| koi_incl_err1 | 9564 | 100 |
| koi_model_dof | 9564 | 100 |
| koi_eccen_err1 | 9564 | 100 |
| koi_model_chisq | 9564 | 100 |
| koi_longp | 9564 | 100 |
| koi_longp_err1 | 9564 | 100 |
| koi_eccen_err2 | 9564 | 100 |
| kepler_name | 6816 | 71.2673 |
| koi_score | 1510 | 15.7884 |
| koi_bin_oedp_sig | 1510 | 15.7884 |
| koi_comment | 1209 | 12.6412 |
| koi_num_transits | 1142 | 11.9406 |
| koi_max_sngle_ev | 1142 | 11.9406 |
| koi_quarters | 1142 | 11.9406 |
| koi_max_mult_ev | 1142 | 11.9406 |
| koi_fwm_stat_sig | 1076 | 11.2505 |
| koi_fwm_prao_err | 830 | 8.67838 |
| koi_fwm_prao | 830 | 8.67838 |
| koi_fwm_pdeco_err | 817 | 8.54245 |
| koi_fwm_pdeco | 817 | 8.54245 |
| koi_zmag | 613 | 6.40945 |
| koi_dicco_mdec | 599 | 6.26307 |
| koi_dicco_msky | 599 | 6.26307 |
| koi_dicco_mra_err | 599 | 6.26307 |
| koi_dicco_mra | 599 | 6.26307 |
| koi_dicco_mdec_err | 599 | 6.26307 |
| koi_dicco_msky_err | 599 | 6.26307 |
| koi_dikco_msky_err | 570 | 5.95985 |
| koi_dikco_msky | 570 | 5.95985 |
| koi_dikco_mra_err | 570 | 5.95985 |
| koi_dikco_mdec_err | 570 | 5.95985 |
| koi_dikco_mdec | 570 | 5.95985 |
| koi_dikco_mra | 570 | 5.95985 |
| koi_fwm_sra_err | 506 | 5.29067 |
| koi_fwm_sdec | 506 | 5.29067 |
| koi_fwm_sdec_err | 506 | 5.29067 |
| koi_fwm_sra | 506 | 5.29067 |
| koi_steff_err2 | 483 | 5.05019 |
| koi_slogg_err1 | 468 | 4.89335 |
| koi_slogg_err2 | 468 | 4.89335 |
| koi_srad_err2 | 468 | 4.89335 |
| koi_smass_err1 | 468 | 4.89335 |
| koi_srad_err1 | 468 | 4.89335 |
| koi_smass_err2 | 468 | 4.89335 |
| koi_steff_err1 | 468 | 4.89335 |
| koi_fwm_sdeco | 455 | 4.75742 |
| koi_fwm_sdeco_err | 455 | 4.75742 |
| koi_fwm_srao_err | 455 | 4.75742 |
| koi_fwm_srao | 455 | 4.75742 |
| koi_duration_err1 | 454 | 4.74697 |
| koi_dor_err2 | 454 | 4.74697 |
| koi_period_err1 | 454 | 4.74697 |
| koi_impact_err2 | 454 | 4.74697 |
| koi_impact_err1 | 454 | 4.74697 |
| koi_duration_err2 | 454 | 4.74697 |
| koi_depth_err2 | 454 | 4.74697 |
| koi_period_err2 | 454 | 4.74697 |
| koi_time0_err1 | 454 | 4.74697 |
| koi_time0bk_err2 | 454 | 4.74697 |
| koi_time0bk_err1 | 454 | 4.74697 |
| koi_time0_err2 | 454 | 4.74697 |
| koi_depth_err1 | 454 | 4.74697 |
| koi_dor_err1 | 454 | 4.74697 |
| koi_smet_err1 | 387 | 4.04642 |
| koi_smet_err2 | 387 | 4.04642 |
| koi_smet | 386 | 4.03597 |
| koi_incl | 364 | 3.80594 |
| koi_prad | 363 | 3.79548 |
| koi_ldm_coeff3 | 363 | 3.79548 |
| koi_impact | 363 | 3.79548 |
| koi_steff | 363 | 3.79548 |
| koi_teq | 363 | 3.79548 |
| koi_depth | 363 | 3.79548 |
| koi_slogg | 363 | 3.79548 |
| koi_smass | 363 | 3.79548 |
| koi_sparprov | 363 | 3.79548 |
| koi_srad | 363 | 3.79548 |
| koi_trans_mod | 363 | 3.79548 |
| koi_ldm_coeff2 | 363 | 3.79548 |
| koi_ldm_coeff1 | 363 | 3.79548 |
| koi_sma | 363 | 3.79548 |
| koi_prad_err1 | 363 | 3.79548 |
| koi_ldm_coeff4 | 363 | 3.79548 |
| koi_limbdark_mod | 363 | 3.79548 |
| koi_ror_err2 | 363 | 3.79548 |
| koi_ror_err1 | 363 | 3.79548 |
| koi_ror | 363 | 3.79548 |
| koi_dor | 363 | 3.79548 |
| koi_prad_err2 | 363 | 3.79548 |
| koi_eccen | 363 | 3.79548 |
| koi_model_snr | 363 | 3.79548 |
| koi_datalink_dvs | 346 | 3.61773 |
| koi_datalink_dvr | 346 | 3.61773 |
| koi_tce_delivname | 346 | 3.61773 |
| koi_tce_plnt_num | 346 | 3.61773 |
| koi_srho_err2 | 321 | 3.35634 |
| koi_insol_err1 | 321 | 3.35634 |
| koi_insol_err2 | 321 | 3.35634 |
| koi_insol | 321 | 3.35634 |
| koi_srho_err1 | 321 | 3.35634 |
| koi_srho | 321 | 3.35634 |
| koi_kmag_err | 193 | 2.01798 |
| koi_imag | 154 | 1.6102 |
| koi_hmag_err | 97 | 1.01422 |
| koi_jmag_err | 72 | 0.752823 |
| koi_gmag | 41 | 0.428691 |
| koi_hmag | 25 | 0.261397 |
| koi_kmag | 25 | 0.261397 |
| koi_jmag | 25 | 0.261397 |
| koi_rmag | 9 | 0.0941029 |
| koi_kepmag | 1 | 0.0104559 |

## Descriptive statistics for required numeric columns

| column | count | mean | std | min | 25% | 50% | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| koi_period | 9564 | 75.6714 | 1334.74 | 0.241843 | 2.73368 | 9.75283 | 40.7152 | 129996 |
| koi_duration | 9564 | 5.62161 | 6.47155 | 0.052 | 2.43775 | 3.7926 | 6.2765 | 138.54 |
| koi_depth | 9201 | 23791.3 | 82242.7 | 0 | 159.9 | 421.1 | 1473.4 | 1.5414e+06 |
| koi_prad | 9201 | 102.892 | 3077.64 | 0.08 | 1.4 | 2.39 | 14.93 | 200346 |
| koi_model_snr | 9201 | 259.895 | 795.807 | 0 | 12 | 23 | 78 | 9054.7 |
| koi_teq | 9201 | 1085.39 | 856.351 | 25 | 539 | 878 | 1379 | 14667 |

Numeric conversion was performed in memory only for this report. The raw CSV remains unchanged.
