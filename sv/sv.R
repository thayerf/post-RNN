data <- read.csv("data.csv", header = FALSE)
labels <- read.csv("labels.csv", header = FALSE)
loss <- read.csv("loss", header = FALSE)
loss <- loss[,1]
labels <- labels[,1]
library(stochvol)
res <- rep(0,500)

for(i in 1:500){
  temp <- svsample(as.numeric(data[i,]),priormu = c(0,10), priorphi = c(7,1.5),priorsigma = 1.5)
  res[i]<-summary(temp,showlatent=FALSE)$para[1,4]
}

sv_loss <- mean(0.5*abs(res-labels))
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

max.epoc <- 100

width <- 2

colors <- c(rgb(56.25/255,34.50/255,113.25/255,0.3), rgb(0,0,1,0.8))

inds.to.use <- round(seq(from = 1, to = max.epoc, length.out = 100))

pdf("sv.pdf", width = 6, height = 4)
plot(inds.to.use,loss, type = 'n', yaxs ='i', xlab = "Epoch", ylab = "risk", ylim = c(0.00,3.0))
lines(inds.to.use,loss, lwd = width, col = colors[1])
abline(h= sv_loss, lwd= width,lty = 2,col = "red")
legend(x = 'topright', legend=c("RNN", "stochvol"),
       col=c(colors[1], "red"), lty=c(1,2), cex=0.8)
dev.off()
