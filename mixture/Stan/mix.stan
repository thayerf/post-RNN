data {
  int<lower = 0> N;
  vector[N] y;
}

parameters {
  vector[5] mu;
}
transformed parameters{
  real theta;
  theta= max(mu);
}

model {
  mu[1] ~ normal(0, 0.1);
  mu[2] ~ normal(0, 0.1);
  mu[3] ~ normal(0, 0.1);
  mu[4] ~ normal(0, 0.1);
  mu[5] ~ normal(0, 0.1);
  for (n in 1:N){
    target += 0.2*normal_lpdf(y[n]| mu[1],1);
    target += 0.2*normal_lpdf(y[n]| mu[2],1);
    target += 0.2*normal_lpdf(y[n]| mu[3],1);
    target += 0.2*normal_lpdf(y[n]| mu[4],1);
    target += 0.2*normal_lpdf(y[n]| mu[5],1);
  }
}
