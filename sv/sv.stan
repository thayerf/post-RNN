data {
  int<lower = 0> N;
  vector[N] y;
}

parameters {
  real mu;
  real phi;
  real sigma;
  vector[N] h; 
}


model {
  mu ~ normal(0,10);
  phi~ beta(7,1.5);
  sigma ~ chi_square(1);
  h[1]~normal(mu,sqrt(1.5*sigma/(1-square(2*phi-1))));
  y[1] ~ normal(0,sqrt(exp(h[1])));
  for(i in 2:(N-1)){
    h[i]~normal(mu+(2*phi-1)*(h[i-1]-mu),sqrt(1.5*sigma));
    y[i]~normal(0,sqrt(exp(h[i])));
  }
}