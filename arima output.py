Python 3.11.3 (tags/v3.11.3:f3909b8, Apr  4 2023, 23:49:59) [MSC v.1934 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.

======================= RESTART: C:/Users/my241/fit2.0.py ======================

📊 Processing stock: AXISBANK
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                  Close   No. Observations:                 2588
Model:                 ARIMA(4, 1, 0)   Log Likelihood              -12330.180
Date:                Tue, 10 Jun 2025   AIC                          24670.360
Time:                        13:43:12   BIC                          24699.651
Sample:                    06-01-2011   HQIC                         24680.976
                         - 04-30-2021                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ar.L1          0.0060      0.055      0.109      0.913      -0.102       0.113
ar.L2         -0.0062      0.054     -0.114      0.909      -0.112       0.100
ar.L3          0.0140      0.047      0.296      0.768      -0.079       0.107
ar.L4          0.0023      0.066      0.035      0.972      -0.127       0.132
sigma2      1361.4669      2.324    585.813      0.000    1356.912    1366.022
===================================================================================
Ljung-Box (L1) (Q):                   1.40   Jarque-Bera (JB):         240340669.97
Prob(Q):                              0.24   Prob(JB):                         0.00
Heteroskedasticity (H):               0.05   Skew:                           -33.66
Prob(H) (two-sided):                  0.00   Kurtosis:                      1494.69
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).

📊 Processing stock: BAJFINANCE

Warning (from warnings module):
  File "C:\Users\my241\AppData\Roaming\Python\Python311\site-packages\statsmodels\base\model.py", line 607
    warnings.warn("Maximum Likelihood optimization failed to "
ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                  Close   No. Observations:                 2588
Model:                 ARIMA(3, 1, 4)   Log Likelihood              -16689.456
Date:                Tue, 10 Jun 2025   AIC                          33394.912
Time:                        13:43:21   BIC                          33441.778
Sample:                    06-01-2011   HQIC                         33411.897
                         - 04-30-2021                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ar.L1          0.3817      2.106      0.181      0.856      -3.746       4.509
ar.L2          0.4097      0.969      0.423      0.672      -1.490       2.309
ar.L3          0.0625      0.104      0.603      0.546      -0.141       0.266
ma.L1         -0.3770      2.104     -0.179      0.858      -4.500       3.746
ma.L2         -0.2917      0.980     -0.298      0.766      -2.212       1.628
ma.L3         -0.4025      0.346     -1.164      0.245      -1.080       0.275
ma.L4          0.2046      0.506      0.404      0.686      -0.787       1.196
sigma2      4.773e+04    490.088     97.397      0.000    4.68e+04    4.87e+04
===================================================================================
Ljung-Box (L1) (Q):                   5.10   Jarque-Bera (JB):         328586649.74
Prob(Q):                              0.02   Prob(JB):                         0.00
Heteroskedasticity (H):              10.48   Skew:                           -37.85
Prob(H) (two-sided):                  0.00   Kurtosis:                      1747.31
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).

📊 Processing stock: BAJAJFINSV
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                  Close   No. Observations:                 2588
Model:                 ARIMA(7, 1, 0)   Log Likelihood              -14847.374
Date:                Tue, 10 Jun 2025   AIC                          29710.748
Time:                        13:43:24   BIC                          29757.614
Sample:                    06-01-2011   HQIC                         29727.733
                         - 04-30-2021                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ar.L1         -0.0245      0.009     -2.640      0.008      -0.043      -0.006
ar.L2         -0.0017      0.011     -0.156      0.876      -0.023       0.020
ar.L3          0.0600      0.009      6.414      0.000       0.042       0.078
ar.L4          0.0571      0.011      4.992      0.000       0.035       0.080
ar.L5          0.0560      0.013      4.355      0.000       0.031       0.081
ar.L6         -0.0616      0.012     -5.033      0.000      -0.086      -0.038
ar.L7          0.0811      0.013      6.105      0.000       0.055       0.107
sigma2      1.069e+04    105.481    101.388      0.000    1.05e+04    1.09e+04
===================================================================================
Ljung-Box (L1) (Q):                   0.29   Jarque-Bera (JB):            100003.21
Prob(Q):                              0.59   Prob(JB):                         0.00
Heteroskedasticity (H):             146.31   Skew:                            -0.59
Prob(H) (two-sided):                  0.00   Kurtosis:                        33.44
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).

📊 Processing stock: HDFCBANK
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                  Close   No. Observations:                 2588
Model:                 ARIMA(1, 1, 0)   Log Likelihood              -13026.748
Date:                Tue, 10 Jun 2025   AIC                          26057.496
Time:                        13:43:26   BIC                          26069.213
Sample:                    06-01-2011   HQIC                         26061.742
                         - 04-30-2021                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ar.L1         -0.0485      0.012     -4.104      0.000      -0.072      -0.025
sigma2      2415.3326      2.924    826.064      0.000    2409.602    2421.063
===================================================================================
Ljung-Box (L1) (Q):                   1.53   Jarque-Bera (JB):         187539768.37
Prob(Q):                              0.22   Prob(JB):                         0.00
Heteroskedasticity (H):               0.39   Skew:                           -33.44
Prob(H) (two-sided):                  0.00   Kurtosis:                      1320.33
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).

📊 Processing stock: ICICIBANK

Warning (from warnings module):
  File "C:\Users\my241\AppData\Roaming\Python\Python311\site-packages\statsmodels\base\model.py", line 607
    warnings.warn("Maximum Likelihood optimization failed to "
ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                  Close   No. Observations:                 2588
Model:                 ARIMA(3, 1, 1)   Log Likelihood              -11985.547
Date:                Tue, 10 Jun 2025   AIC                          23981.095
Time:                        13:43:29   BIC                          24010.386
Sample:                    06-01-2011   HQIC                         23991.710
                         - 04-30-2021                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ar.L1          0.9675      0.054     17.879      0.000       0.861       1.074
ar.L2          0.0141      0.036      0.396      0.692      -0.056       0.084
ar.L3          0.0084      0.033      0.258      0.796      -0.055       0.072
ma.L1         -0.9933      0.052    -19.271      0.000      -1.094      -0.892
sigma2      1027.6628      5.696    180.408      0.000    1016.498    1038.827
===================================================================================
Ljung-Box (L1) (Q):                   1.26   Jarque-Bera (JB):         315410152.55
Prob(Q):                              0.26   Prob(JB):                         0.00
Heteroskedasticity (H):               0.22   Skew:                           -37.26
Prob(H) (two-sided):                  0.00   Kurtosis:                      1711.96
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).

📊 Processing stock: INDUSINDBK
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                  Close   No. Observations:                 2588
Model:                 ARIMA(7, 1, 3)   Log Likelihood              -10872.153
Date:                Tue, 10 Jun 2025   AIC                          21766.305
Time:                        13:43:34   BIC                          21830.746
Sample:                    06-01-2011   HQIC                         21789.660
                         - 04-30-2021                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ar.L1          0.5244      0.186      2.823      0.005       0.160       0.888
ar.L2         -0.1120      0.219     -0.512      0.608      -0.541       0.317
ar.L3          0.1566      0.172      0.911      0.362      -0.180       0.493
ar.L4          0.0276      0.019      1.428      0.153      -0.010       0.066
ar.L5         -0.0795      0.018     -4.391      0.000      -0.115      -0.044
ar.L6          0.0091      0.023      0.387      0.699      -0.037       0.055
ar.L7          0.0822      0.022      3.782      0.000       0.040       0.125
ma.L1         -0.4641      0.186     -2.490      0.013      -0.829      -0.099
ma.L2          0.0918      0.209      0.439      0.661      -0.318       0.502
ma.L3         -0.1608      0.164     -0.978      0.328      -0.483       0.161
sigma2       411.0924      5.837     70.429      0.000     399.652     422.533
===================================================================================
Ljung-Box (L1) (Q):                   0.01   Jarque-Bera (JB):              7271.24
Prob(Q):                              0.93   Prob(JB):                         0.00
Heteroskedasticity (H):              12.56   Skew:                            -0.05
Prob(H) (two-sided):                  0.00   Kurtosis:                        11.21
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).

📊 Processing stock: KOTAKBANK
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                  Close   No. Observations:                 2588
Model:                 ARIMA(3, 1, 1)   Log Likelihood              -11303.907
Date:                Tue, 10 Jun 2025   AIC                          22617.814
Time:                        13:43:36   BIC                          22647.105
Sample:                    06-01-2011   HQIC                         22628.430
                         - 04-30-2021                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ar.L1         -0.0127      0.734     -0.017      0.986      -1.451       1.426
ar.L2         -0.0218      0.026     -0.849      0.396      -0.072       0.029
ar.L3          0.0235      0.018      1.274      0.203      -0.013       0.060
ma.L1         -0.0172      0.738     -0.023      0.981      -1.464       1.430
sigma2       588.8350      1.721    342.191      0.000     585.462     592.208
===================================================================================
Ljung-Box (L1) (Q):                   0.74   Jarque-Bera (JB):          12352929.80
Prob(Q):                              0.39   Prob(JB):                         0.00
Heteroskedasticity (H):               5.74   Skew:                           -10.95
Prob(H) (two-sided):                  0.00   Kurtosis:                       340.82
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).

📊 Processing stock: SBIN
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                  Close   No. Observations:                 2588
Model:                 ARIMA(3, 1, 1)   Log Likelihood              -13454.065
Date:                Tue, 10 Jun 2025   AIC                          26918.131
Time:                        13:43:39   BIC                          26947.422
Sample:                    06-01-2011   HQIC                         26928.747
                         - 04-30-2021                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ar.L1          0.2165      0.729      0.297      0.766      -1.212       1.645
ar.L2         -0.0104      0.045     -0.230      0.818      -0.099       0.078
ar.L3         -0.0447      0.016     -2.754      0.006      -0.077      -0.013
ma.L1         -0.2118      0.723     -0.293      0.770      -1.630       1.206
sigma2      3416.7032      8.600    397.273      0.000    3399.847    3433.560
===================================================================================
Ljung-Box (L1) (Q):                   1.60   Jarque-Bera (JB):         314721639.71
Prob(Q):                              0.21   Prob(JB):                         0.00
Heteroskedasticity (H):               0.03   Skew:                           -37.25
Prob(H) (two-sided):                  0.00   Kurtosis:                      1710.09
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
