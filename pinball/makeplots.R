
n.60.loss <- read.table("output/nn-n-60-p-2-1000-hidden.out")[,1]

max.epoc <- length(n.60.loss)

width <- 2

colors <- c(rgb(56.25/255,34.50/255,113.25/255,0.3), rgb(0,0,1,0.8))

inds.to.use <- round(seq(from = 1, to = max.epoc, length.out = 200))

pdf("deep-learning.pdf", width = 6, height = 4)
plot(inds.to.use/1000,n.60.loss[inds.to.use], type = 'n', yaxs ='i', xlab = "iteration # (in thousands)", ylab = "risk", log="y", ylim = c(0.05,1.5))
  lines(inds.to.use/1000,n.60.loss[inds.to.use], lwd = width, col = colors[1])
abline(h = 0.38, lty = 2)
dev.off()


