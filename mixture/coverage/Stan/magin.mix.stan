data {
  int<lower = 0> N;
  vector[N] y;
  int<lower = 0> k;
  vector[11]
}

parameters {
  vector[k] mu;
}
transformed parameters{
  real theta;
  theta= max(mu);
}

model {
  vector[k] delta;
  for(i in 1:k){
    mu[i] ~ normal(0, 0.707);
  }
  for (n in 1:N){
    for(j in 1:k){
      delta[j] = normal_lpdf(y[n]| mu[j],1.0);
    }
    target += log_sum_exp(delta);
  }
}
