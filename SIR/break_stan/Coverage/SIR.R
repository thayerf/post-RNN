I <- read.csv("I_data.csv", header = FALSE)
S <- read.csv("S_data.csv",header = FALSE)
labels <- read.csv("labels.csv", header = FALSE)
loss <- read.csv("loss", header = FALSE)
loss <- loss[,1]
labels <- labels[,1]
preds <- read.csv('preds', header = FALSE)[,1]
library(rstan)
losses <- 0.5*abs(preds-labels)
pred_se <- sd(losses)/sqrt(length(losses))
res <- rep(0,2000)
for(i in 1:2000){
  input_data <- c()
  input_data$S <- as.numeric(S[i,-1])
  input_data$I <- as.numeric(I[i,-1])
  input_data$T <- 5
  my_model = stan(file = "SIR.stan", data=input_data,seed=1408)
  res[i]<-summary(my_model)$summary[3,5]
}

stan_loss <- mean(0.5*abs(res[1:1745]-labels[1:1745]))


max.epoc <- 2000

width <- 2

colors <- c(rgb(56.25/255,34.50/255,113.25/255,0.3), rgb(0,0,1,0.8))

inds.to.use <- round(seq(from = 1, to = max.epoc, length.out = 300))*100
pdf("sir.pdf", width = 6, height = 4)
plot(inds.to.use,loss, type = 'n', yaxs ='i', xlab = "Number of Simulated Datasets", ylab = "risk", ylim = c(0.00,2.0))
lines(inds.to.use,predict(loess(loss~c(1:300), span = 0.1)), lwd = width, col = colors[1])
abline(h= stan_loss, lwd= width,lty = 2, col = "red")
legend(x = 'topright', legend=c("RNN", "Stan"),
       col=c(colors[1],"red"), lty=c(1,2), cex=0.8)
arrows(max(inds.to.use), tail(loss, n=1)-1.96*pred_se,max(inds.to.use),tail(loss, n=1)+1.96*pred_se,length=0.05, angle=90, code=3)

dev.off()
