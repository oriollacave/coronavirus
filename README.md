# coronavirus data process and plots
read coronavirus data from John Hopkins & ERA5 reanalysis dataset for meteo.
Plots different countries plots of pandemia evolution.
Uses a)hipothesis [1% death rate] to compute infected estimates.
Uses b)hipotesis [7 days delay average between official detections and deaths] to use infection data to predict deaths. This also uses previous hipotesis for further predictors in addition to infections.
Computes R0 spread ratio using infections & deaths.
Compute detection ratio based on hipotesis a).
Plots raw accumulated data vs time. Plots daily diffs vs time. Plots R0 vs time. Plots previous estimates [real infections and deaths forecast] vs time.
Compute and plot P94-P98 max R0 values average    VS     detectionRatio with temperature in colors.
     --> this shows that both detection ratio and temperature can itself contain R0 at reasonable levels (despite not being <=1)
