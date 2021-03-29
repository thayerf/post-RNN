data {
  int<lower = 0> N;
  vector[N] y;
  vector[11] weights;
}

parameters {
  vector[11] mu;
}
transformed parameters{
  real theta;
  for(i in 1:11){
    theta += weights[i]*max(mu[1:i]);
  }
}

model {
  vector[11] delta;
  for(i in 1:11){
    mu[i] ~ normal(0, 0.707);
  }
  for (n in 1:N){
    for(i in 1:11){
    for(j in 1:i){
      delta[j] = log(weights[i])+log(1.0/j)+normal_lpdf(y[n]| mu[j],1.0);
    }
    target += log_sum_exp(delta[1:i]);
    }
  }
}
