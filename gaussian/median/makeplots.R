library(tidyverse)
loss <- read_csv("loss", col_names = FALSE)
labels <- read_csv("labels.csv", col_names = FALSE)
data <- read_csv("data.csv", col_names = FALSE)
preds <- read_csv("average_preds", col_names = FALSE)
loss <- loss[,1][[1]]
labels <- labels[,1]
preds <- preds[,1][[1]]
max.epoc <- 2000
sig <- 1/100
post_med <- 0.5*rowMeans(data)
post_loss <- 0.5*mean(abs(post_med-labels[[1]]))
width <- 2
pred_se <- sd(preds)/sqrt(length(preds))





colors <- c(rgb(56.25/255,34.50/255,113.25/255,0.3), rgb(0,0,1,0.8))

inds.to.use <- round(seq(from = 1, to = max.epoc, length.out = 2000))*1000
# Uncomment to save pdf
#pdf("gauss.pdf", width = 6, height = 4)
plot(inds.to.use,log(loss), type = 'n', yaxs ='i', xlab = "Number of Simulated Datasets", ylab = "Standardized Risk", ylim = c(0,10.0))
lines(inds.to.use,predict(loess(loss/post_loss~c(1:max.epoc), span = 0.1)), lwd = width, col = colors[1])

abline(h= 1, lwd= width,lty = 2, col = "red")

legend(x = 'topright', legend=c("RNN", "Posterior Median"),
       col=c(colors[1],"red"), lty=c(1,2), cex=0.8)

arrows(max(inds.to.use), (tail(loss, n=1)-1.96*pred_se)/post_loss,max(inds.to.use),(tail(loss, n=1)+1.96*pred_se)/post_loss,length=0.05, angle=90, code=3)
#dev.off()
