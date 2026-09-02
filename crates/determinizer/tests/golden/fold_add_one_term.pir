(%module
  (%public g obs lk lp)

  (%bind g 0.0)

  (%bind obs (%meta ((%array 1 (1) (%scalar real)) %fixed (cartpow (interval 0.0 0.0) 1)) (vector 0.0)))

  (%bind lk 0.0)

  (%bind lp (%meta ((%scalar real) %fixed reals) (builtin_logdensityof Normal (%meta ((%record (mu (%scalar real)) (sigma (%scalar real))) %fixed (record (mu reals) (sigma reals))) (record (%field mu 0.0) (%field sigma 1.0))) (%meta ((%scalar real) %fixed reals) (get0 (%ref self obs) 0))))))