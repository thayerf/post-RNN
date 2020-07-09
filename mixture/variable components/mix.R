labels <- read.csv('labels.csv',header = FALSE)
labels <- labels[,]
data <- read.csv('data.csv', header = FALSE)
data <- data[,]
loss <- read.csv('loss',header= F)
julia <- as.numeric(read.csv('juliapreds.csv')[1,])
rnn_preds <- read.table('average_preds')[,]

rnn_mad <- abs(rnn_preds-labels)

cor(rnn_preds, labels)

data <- as.matrix(data)

losses <- abs(rnn_preds-labels)
pred_se <- sd(losses)/sqrt(length(losses))
# Plug in test data
library(mclust)
comps <- rep(0, 500)
for(i in 1:nrow(data)){
  comps[i] = which.max(Mclust(data[i,], G = 1:9, model = "E")$BIC)
}

preds05 <- c()
preds50 <- c()
preds95 <- c()
library(rstan)
input_data <- c()
for(i in 1:500){
input_data$y <- as.numeric(data[i,])
input_data$N <- 250
input_data$k <- comps[i]
my_model = stan(file = "mix.stan", data=input_data,seed=1408)
preds05 <- append(preds05,quantile(extract(my_model)$theta,0.05))
preds50 <- append(preds50,quantile(extract(my_model)$theta,0.5))
preds95 <- append(preds95,quantile(extract(my_model)$theta,0.95))
}


prior_mean05 <- c()
prior_mean50 <- c()
prior_mean95 <- c()
for(i in 1:500){
  input_data$y <- as.numeric(data[i,])
  input_data$N <- 250
  input_data$k <- 5
  my_model = stan(file = "mix.stan", data=input_data,seed=1408)
  prior_mean05 <- append(prior_mean05,quantile(extract(my_model)$theta,0.05))
  prior_mean50 <- append(prior_mean50,quantile(extract(my_model)$theta,0.5))
  prior_mean95 <- append(prior_mean95,quantile(extract(my_model)$theta,0.95))
}



margin_mean05 <- c()
margin_mean50 <- c()
margin_mean95 <- c()
for(i in 1:500){
  input_data$y <- as.numeric(data[i,])
  input_data$N <- 250
  input_data$weights <- dpois(1:11,5)/sum(dpois(1:11,5))
  my_model = stan(file = "margin.mix.stan", data=input_data,seed=1408)
  margin_mean05 <- append(margin_mean05,quantile(extract(my_model)$theta,0.05))
  margin_mean50 <- append(margin_mean50,quantile(extract(my_model)$theta,0.5))
  margin_mean95 <- append(margin_mean95,quantile(extract(my_model)$theta,0.95))
}



stan_mad <- mean(abs(preds50-labels))

stan_prior_mad <- mean(abs(prior_mean50-labels))
rnn_losses <- c()
rnn_losses$loss <- c(mean(abs(labels)),2*loss[,1])
rnn_losses$index <- 1:length(rnn_losses$loss)
lo10 <- loess(loss~index,data= rnn_losses, span = 0.1)
plot(lo10, type="l", col="red", lwd=5, xlab="Epoch", ylab="Loss", main="Neural Net vs. BIC + STAN",
     ylim = range(0:1))
abline(h= stan_mad, lwd= 3,lty = 2)
abline(h= stan_prior_mad, lwd = 3, lty = 2, col = 'blue')
abline(h = mean(abs(labels-julia)), col = 'green', lwd = 3, lty= 2)
legend(x = 'topright', legend=c("RNN", "BIC + STAN", "Julia", "Prior Mean # of Clusters + STAN"),
       col=c("red", "black","green","blue"), lty=c(1,2,2), cex=0.8)
load("stan_mad")
load("stan_prior_mad")




max.epoc <- 201

width <- 2

colors <- c(rgb(56.25/255,34.50/255,113.25/255,0.3), rgb(0,0,1,0.8))

inds.to.use <- round(seq(from = 1, to = max.epoc, length.out = 201))*150

pdf("mfm.pdf", width = 6, height = 4)
plot(inds.to.use,rnn_losses$loss, type = 'n', yaxs ='i',
     xlab = "Number of Simulated Datasets", ylab = "risk", ylim = c(0.00,0.80))
lines(inds.to.use,rnn_losses$loss, lwd = width, col = colors[1])
abline(h= stan_mad, lwd= width,lty = 2)
abline(h= stan_prior_mad, lwd = width, lty = 2, col = 'blue')
abline(h = mean(abs(labels-julia)), col = 'red', lwd = width, lty= 2)
legend(x = 'topright', legend=c("RNN", "BIC + STAN", "Julia", "Prior Mean # of Clusters + STAN"),
       col=c(colors[1], "black","red","blue"), lty=c(1,2,2,2), cex=0.8)
arrows(max(inds.to.use), 2*tail(loss[,1], n=1)-1.96*pred_se,max(inds.to.use),2*tail(loss[,1], n=1)+1.96*pred_se,length=0.05, angle=90, code=3)
dev.off()