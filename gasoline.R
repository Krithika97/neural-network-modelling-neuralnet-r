#1. SET DIRECTORY AND FILE PATH
setwd("directory")
fullData <- read.csv("gasoline.csv")
attach(fullData)
fullData<-data.frame(consumption, capacity, gasoline, hours)
attach(fullData)

#2. MAX-MIN NORMALIZATION
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
maxmindf <- as.data.frame(lapply(fullData, normalize))

#3.TRAINING AND TEST DATA
trainset <- maxmindf[1:32, ]
testset <- maxmindf[33:40, ]

#4. NEURAL NETWORK
library(neuralnet)
nn <- neuralnet(consumption ~ capacity + price + hours,data=trainset, hidden=c(5,2), linear.output=TRUE, threshold=0.01)
nn$result.matrix
plot(nn)
temp_test <- subset(testset, select = c("capacity","price","hours"))
head(temp_test)
nn.results <- compute(nn, temp_test)

#5. MODEL VALIDATION
results <- data.frame(actual = testset$consumption, prediction = nn.results$net.result)
results

predicted=results$prediction * abs(diff(range(consumption))) + min(consumption)
actual=results$actual * abs(diff(range(consumption))) + min(consumption)
comparison=data.frame(predicted,actual)
deviation=((actual-predicted)/actual)
comparison=data.frame(predicted,actual,deviation)
accuracy=1-abs(mean(deviation))
accuracy
1-mean(abs(deviation))
