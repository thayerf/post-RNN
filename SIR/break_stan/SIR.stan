data {
  int<lower=0> T;   // # time points (equally spaced)
  vector[T] S;      // Suseptible people at time t
  vector[T] I;      // Infected people at time t
}
parameters {
  real<lower = 0> beta;                     // infection rate
  real <lower = 0> gamma;                   // recovery rate
}
transformed parameters{
  real <lower = 0> r = beta/gamma; // Mean number of infections an individual transmits to a fully Suseptible pop before recovering
}
model {
  real S_0 = 98;
  real I_0 = 2; // Initial number of infected is fixed here, but you could put a prior on this if you wanted to
  real N = 100;
  beta ~ gamma(9, 20);
  gamma ~ gamma(3, 20);
  S[1] ~ normal((S_0-20*beta*S_0*I_0/N),sqrt(20*beta*S_0*I_0/N));
  I[1] ~ normal(I_0+S_0-S[1]-20*gamma*I_0,sqrt(20*gamma*I_0));
  for (t in 2:T){
    if(S[t-1]>0 && I[t-1]>0){
      S[t] ~ normal(S[t-1]-20*beta*S[t-1]*I[t-1]/N,sqrt(20*beta*S[t-1]*I[t-1]/N));
      I[t] ~ normal(I[t-1]+S[t-1]-S[t]-20*gamma*I[t-1],sqrt(20*gamma*I[t-1]));
    }
    else{
      S[t] ~ normal(S[t-1], 1e-10);
      I[t] ~ normal(I[t-1], 1e-10);
    }
    
  }
}