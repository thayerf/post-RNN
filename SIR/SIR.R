setwd("C:/Users/thaye_000/Dropbox/Noah and Thayer/RNN/SIR")

I <- read.csv("data/I_data.csv", header = FALSE)
S <- read.csv("data/S_data.csv",header = FALSE)
labels <- read.csv("data/labels.csv", header = FALSE)
loss <- read.csv("loss", header = FALSE)
loss <- loss[,1]
labels <- labels[,1]

library(rstan)

res <- rep(0,2000)
for(i in 1:2000){
  input_data <- c()
  input_data$S <- as.numeric(S[i,-1])
  input_data$I <- as.numeric(I[i,-1])
  input_data$T <- 99
  my_model = stan(file = "SIR.stan", data=input_data,seed=1408)
  res[i]<-summary(my_model)$summary[3,5]
}

stan_loss <- mean(0.5*abs(res-labels))
preds <- read.csv("preds", header = FALSE)
preds <- preds[,1]




preds05 <- c()
preds50 <- c()
preds95 <- c()
library(rstan)
input_data <- c()
for(i in 1:500){
input_data$y <- as.numeric(data[i,])
input_data$N <- 75
my_model = stan(file = "sv.stan", data=input_data,seed=1408)
preds05 <- append(preds05,quantile(extract(my_model)$mu,0.05))
preds50 <- append(preds50,quantile(extract(my_model)$mu,0.5))
preds95 <- append(preds95,quantile(extract(my_model)$mu,0.95))
}

max.epoc <- 400

width <- 2

colors <- c(rgb(56.25/255,34.50/255,113.25/255,0.3), rgb(0,0,1,0.8))

inds.to.use <- round(seq(from = 1, to = max.epoc, length.out = 400))
pdf("sir.pdf", width = 6, height = 4)
plot(inds.to.use,loss, type = 'n', yaxs ='i', xlab = "Epoch", ylab = "risk", ylim = c(0.00,2.0))
lines(inds.to.use,predict(loess(loss~c(1:400), span = 0.1)), lwd = width, col = colors[1])
abline(h= stan_loss, lwd= width,lty = 2, col = "red")
legend(x = 'topright', legend=c("RNN", "Stan"),
       col=c(colors[1],"red"), lty=c(1,2), cex=0.8)

dev.off()
