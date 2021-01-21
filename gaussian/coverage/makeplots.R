library(tidyverse)
loss <- read_csv("results/loss", col_names = FALSE)
labels <- read_csv("results/labels.csv", col_names = FALSE)
data <- read_csv("results/data.csv", col_names = FALSE)
preds <- read_csv("results/preds", col_names = FALSE)
loss <- loss[,1][[1]]
labels <- labels %>% pull(X1)
preds <- preds[,1][[1]]
max.epoc <- 2000
sig <- 1/100
post_mean <- 0.5*rowMeans(data)
post_var <- 1/200

post_025 <- qnorm(0.025, mean = post_mean, sd = sqrt(post_var))
post_05 <- qnorm(0.05, mean = post_mean, sd = sqrt(post_var))
post_95 <- qnorm(0.95, mean = post_mean, sd = sqrt(post_var))


pred_05 <-as.numeric(unlist(strsplit(preds[1],' ')))
pred_95 <- as.numeric(unlist(strsplit(preds[2],' ')))
















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
