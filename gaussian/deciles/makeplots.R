loss <- read.csv("loss")
labels <- read.csv("labels.csv", header = FALSE)
data <- read.csv("data.csv", header = FALSE)
preds <- as.matrix(read.delim("preds", header = FALSE, sep = " "))
loss <- loss[,1]
labels <- labels[,1]
max.epoc <- 299
sig <- 1/100
post_med <- 0.5*rowMeans(data)
quants = matrix(nrow = 9, ncol = 500)
losses = matrix(nrow = 9, ncol = 500)
pred_losses = matrix(nrow = 9, ncol = 500)

for(i in 1:9){
  quants[i,] <- post_med + qnorm(0.1 * i)/sqrt(200)
  losses[i,] <- (1-0.1 * i) * pmax(quants[i,]-labels,0)+(0.1*i)*pmax(labels - quants[i,],0)
  pred_losses[i,] <- (1-0.1 * i) * pmax(preds[i,]-labels,0)+(0.1*i)*pmax(labels - preds[i,],0)
}
dec_losses <- colSums(pred_losses)
pred_se = sd(dec_losses)/sqrt(length(dec_losses))
post_loss = sum(rowMeans(losses))
width <- 2

colors <- c(rgb(56.25/255,34.50/255,113.25/255,0.3), rgb(0,0,1,0.8))

inds.to.use <- round(seq(from = 1, to = max.epoc, length.out = 299))*100
pdf("gauss_dec.pdf", width = 6, height = 4)
plot(inds.to.use,loss, type = 'n', yaxs ='i', xlab = "Number of Simulated Datasets", ylab = "Standardized Risk", ylim = c(0.00,10.0))
lines(inds.to.use,predict(loess(loss/post_loss~c(1:299), span = 0.1)), lwd = width, col = colors[1])
abline(h= 1.0, lwd= width,lty = 2, col = "red")
legend(x = 'topright', legend=c("RNN", "Posterior Deciles"),
       col=c(colors[1],"red"), lty=c(1,2), cex=0.8)
arrows(max(inds.to.use), (tail(loss, n=1)-1.96*pred_se)/post_loss,max(inds.to.use),(tail(loss, n=1)+1.96*pred_se)/post_loss,length=0.05, angle=90, code=3)
dev.off()





