mydata <- read.csv("directory/dividendinfo.csv")
attach(mydata)

#Scaled Normalization
scaleddata<-scale(mydata)

#Max-Min Normalization
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

maxmindf <- as.data.frame(lapply(mydata, normalize))
attach(maxmindf)

#Training and Validation Data
trainset <- maxmindf[1:160, ]
valset <- maxmindf[161:200, ]

#Neural Network
library(neuralnet)
nn <- neuralnet(dividend ~ fcfps + earnings_growth + de + mcap + current_ratio, data=trainset, hidden=c(2,1), linear.output=FALSE, threshold=0.01)
nn$result.matrix
plot(nn)

#Test the resulting output
temp_test <- subset(valset, select = c("fcfps","earnings_growth", "de", "mcap", "current_ratio"))
head(temp_test)
nn.results <- compute(nn, temp_test)

#Accuracy
results <- data.frame(actual = valset$dividend, prediction = nn.results$net.result)
results
roundedresults<-sapply(results,round,digits=0)
roundedresultsdf=data.frame(roundedresults)
attach(roundedresultsdf)
table(actual,prediction)
