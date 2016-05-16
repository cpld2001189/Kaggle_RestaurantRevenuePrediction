#Load all necessary packages installed
library(DMwR)
library(randomForest)
library(caret)
library(dummies)
library(corrplot)
library(ggplot2)
library(reshape)

#The following codes are used for generating features

#FE1: Data cleaning

train <- read.csv("J:\\train.csv")		#read the train CSV file

#FE1.1: Remove outliers of revenue from training data
revenueMatrix <- subset(train, select = c(Id, revenue))
outliers.scores <- lofactor(revenueMatrix, k=15)	#LOF
outliers <- order(outliers.scores, decreasing = T)[1]
print(outliers)
train <- train[-(outliers), ]                   #remove outliers in

n.train <- nrow(train)
test  <- read.csv("J:\\test.csv")               #read test CSV file 
test$revenue <- 1

myData <- train             	#used when cross-validation
myData <- rbind(train, test)   	#used when modelling and prediction

#FE1.2: Remove loosely-correlated variables
myData <- subset(myData, select = -c(City, City.Group, P22))

#FE2: Transformation of attributes

#FE2.1: change MB to DT
myData$Type[myData$Type=="MB"] <- "DT"  
myData$Type <- as.factor(myData$Type)

#Calculate 'lasting days' until 1st Jan 2015
myData$year <- substr(as.character(myData$Open.Date),7,10)
myData$month <- substr(as.character(myData$Open.Date),1,2)
myData$day <- substr(as.character(myData$Open.Date),4,5)
myData$Date <- as.Date(strptime(myData$Open.Date, "%m/%d/%Y"))
myData$days <- as.numeric(as.Date("2015-01-01")-myData$Date)
myData<-subset(myData, select= -c(year, month, day, Date, Open.Date))

#FE2.2: SqrtRoot-Log transform of revenue & lasting days
myData$revenue <- sqrt(log(myData$revenue))
myData$days <- sqrt(log(myData$days))


#FE3: Convert P-Variables to dummies

#Convert P-Variables to dummies
myData <- dummy.data.frame(myData, names=c("P14", "P15", "P16", "P17", "P18", "P24", "P25", "P26", "P27", "P30", "P31", "P32", "P33", "P34", "P35", "P36", "P37"), all=T)

#Remove 0-indicator
myData <- subset(myData, select = -c(P140, P150, P160, P170, P180, P240, P250, P260, P270, P300, P310, P320, P330, P340, P350, P360, P370))

#Convert other P-Variables to dummies
myData <- dummy.data.frame(myData, names=c("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11", "P12", "P13", "P19", "P20", "P21", "P23", "P28", "P29"), all=T)

#Remove '0' indicator for P3 & P29
myData <- subset(myData, select = -c(P30, P290))

#---------------------------------------------------------#
#The following codes are used after features are generated

#RandomForest 10-fold CV
#Only when ‘train’ is assign to ‘myData’
modelCV <- train(revenue~., data=myData, method = "rf",
             trControl=trainControl(method="cv", number=10),
             prox = TRUE, allowParallel = TRUE)
print(modelCV)
print(modelCV$finalModel)


#Random Forest Modelling
#Change the last 4 parameters values if necessary
set.seed(24601)
model <- randomForest(revenue~., 
data=myData[1:n.train,], importance=TRUE, 
mtry = 139, ntree=73500, nPerm=40, nodesize=17)  

#Make a Prediction
prediction <- predict(model, myData[-c(1:n.train), ])

#Back-transform of revenue & write the prediction output to Excel CSV
Submit <- as.data.frame(cbind(seq(0, length(prediction) - 1, by=1),
exp(prediction^2)))
colnames(submit)<-c("Id","Prediction")
write.csv(submit,"winningSoulution_v2_4th.csv",row.names=FALSE,
quote=FALSE)

#---------------------------------------------------------#
#The following codes are for plotting graph

#Plot outliers score
plot(outliers.scores)                           

#Plot graph of variables correlation by hierarchcial clustering
#Convert City, City Group & Type to numeric values
train$City <- as.numeric(train$City)
train$City.Group <- as.numeric(train$City.Group)
train$Type <- as.numeric(train$Type)
numPVar <- sapply(train, is.numeric)
correlation <- cor(train[, numPVar])
corrplot(correlation, order = "hclust")

plot(train$Type)    #plot Type in train
plot(test$Type)     #plot Type in test

hist(train$revenue)             #histogram of revenue
hist(log(train$revenue)         #histogram of Log of revenue
hist(sqrt(train$revenue)        #histogram of SqrtRoot of revenue
hist(sqrt(log(train$revenue)))  #histogram of SqrtRoot-Log of revenue

colnames(myData)                            #check column number
#Histogram of P-Variable Cluster P14-P18, P24-P27, P30-P37
zeroVar <- c(1, 17:21, 26:29, 32:39)    #column number
clusterA <- melt(myData[, zeroVar], id.vars="Id")
ggplot(clusterA, aes( x = value)) + facet_wrap(~variable, scales = "free_x") + geom_histogram()

#histogram of P-Variable Cluster P1-P13, P19-P21, P23, P28-P29
otherVar <- c(1, 4:16, 22:25, 30:31) #column number
clusterB <- melt(myData[, otherVar], id.vars="Id")
ggplot(clusterB, aes( x = value)) + facet_wrap(~variable, scales = "free_x") + geom_histogram() 