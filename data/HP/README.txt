# Housing price data with only 150 samples and some features

data<-read.csv("kaggle_form.csv")
set.seed(0)
factor_variables<-which(sapply(data[1,],class)=="factor")
data_preprocessed<-data[,-factor_variables]
d <- data_preprocessed
d <- data_preprocessed[,c(-1, -2, -3, -8, -9, -26) ]
d <- d[, -6:-31]
d <- d[1:150,]
d[,6] <- - d[,6]
write.csv(d, 'raw', row.names=F)
