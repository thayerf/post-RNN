labels <- read.csv('labels.csv',header = FALSE)
labels <- labels[,]
data <- read.csv('data.csv', header = FALSE)

loss <- read.csv('average_loss',header= F)
library(rstan)
rstan_options(auto_write = TRUE)
N <- 250
n <- 500
preds05 <- c()
preds50 <- c()
preds95 <- c()
input_data <- c()
for(i in 1:n){
input_data$y <- as.numeric(data[i,])
input_data$N <- N

my_model = stan(file='mix.stan', data=input_data,iter = 1000,
                       chains=2, seed=1408, refresh=1000)
preds05 <- append(preds05,quantile(extract(my_model)$theta,0.05))
preds50 <- append(preds50,quantile(extract(my_model)$theta,0.5))
preds95 <- append(preds95,quantile(extract(my_model)$theta,0.95))
}

preds50 <- as.numeric(preds50)
mad <- abs(preds50-labels[1:n])
mean(mad)

rnn_preds <- read.table('average_preds')[,]

rnn_mad <- abs(rnn_preds-labels)

cor(rnn_preds, labels)


cos_sim <- function(x,y){
  x_t <- x/(sqrt(sum(x^2)))
  y_t <- y/(sqrt(sum(y^2)))
  return(sum(x_t*y_t))
}
