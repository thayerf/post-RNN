loss <- read_csv("loss", col_names = FALSE) %>% pull(X1)
rnn_loss <- read_csv("rnn_loss", col_names = FALSE) %>%pull(X1)

loss_inds <- seq(from = 1, to = 200*150*150, length.out = length(loss))

rnn_loss_inds <- seq(from = 1, to = 200*150*150, length.out = length(rnn_loss))

plot(rnn_loss_inds,rnn_loss, type = 'n', yaxs ='i', xlab = "Number of Simulated Datasets", ylab = "Standardized Risk", ylim = c(0.75,3))
lines(loss_inds,predict(loess(loss/min(rnn_loss)~loss_inds, span = 0.1)), lwd = 2, col = 'black')

lines(rnn_loss_inds,predict(loess(rnn_loss/min(rnn_loss)~rnn_loss_inds, span = 0.1)), lwd = 2, col = 'red')