loss <- read.csv("loss")
labels <- read.csv("labels.csv", header = FALSE)
data <- read.csv("data.csv", header = FALSE)
quants <- read.csv("quants.csv", header = FALSE)
preds <- read.csv("preds",header = FALSE)
loss <- loss[,1]
labels <- labels[,1]
quants <- quants[,1]
preds <- preds[,1]
max.epoc <- 299
sig <- 1/100
post_med <- 0.5*rowMeans(data)
quant = as.numeric(rep(0,1500))
losses = as.numeric(rep(0,1500))
pred_losses = as.numeric(rep(0,1500))
for(i in 1:1500){
  quant[i] <- post_med[i] + qnorm(quants[i])/sqrt(200)
  losses[i] <- (1-quants[i]) * pmax(quant[i]-labels[i],0)+(quants[i])*pmax(labels[i] - quant[i],0)
  pred_losses[i] <- (1-quants[i]) * pmax(preds[i]-labels[i],0)+(quants[i])*pmax(labels[i] - preds[i],0)
}
post_loss = mean(losses)
width <- 2
pred_se = sd(pred_losses)/sqrt(length(pred_losses))

colors <- c(rgb(56.25/255,34.50/255,113.25/255,0.3), rgb(0,0,1,0.8))

inds.to.use <- round(seq(from = 1, to = max.epoc, length.out = 299))*100
pdf("gauss_cont.pdf", width = 6, height = 4)
plot(inds.to.use,loss[1:299], type = 'n', yaxs ='i', xlab = "Number of Simulated Datasets", ylab = "Standardized Risk", ylim = c(0.00,10))
lines(inds.to.use,predict(loess(loss[1:299]/post_loss~c(1:299), span = 0.1)), lwd = width, col = colors[1])
abline(h= 1, lwd= width,lty = 2, col = "red")
legend(x = 'topright', legend=c("RNN", "Posterior Quantiles"),
       col=c(colors[1],"red"), lty=c(1,2), cex=0.8)
arrows(max(inds.to.use), (tail(loss, n=1)-1.96*pred_se)/post_loss,max(inds.to.use),(tail(loss, n=1)+1.96*pred_se)/post_loss,length=0.05, angle=90, code=3)
dev.off()





