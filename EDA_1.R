
##################### Exploratory Data Analysis #####################################

## Analysis of Indian Liver Patient Dataset: Classification Problem ##

rm(list = ls())

setwd("C:/Users/vamsi/Desktop/UNL/Pet_Projects/indian_liver-patient")
train.data = read.csv("indian_liver_patient_train.csv")

#Caterogry 1 corresponds to Disease and 2 to No Disease. Renaming to 0 for No Disease
train.data$Category[which(train.data$Category==2)] = 0
train.data$Category = as.factor(train.data$Category) #Converting response to a factor variable


library(tidyverse)
library(ggplot2)

#Looking into the data in details

summary(train.data) 
str(train.data)

na.obs = apply(train.data, 2, function(x) sum(is.na(x))) #Checking if there are any NAs in the dataset
which(is.na(train.data$Albumin_and_Globulin_Ratio)) #Only Albumin_and_Globulin_Ratio has 3 NAs.


# Visualizing each variable for insight into the data

### Categorical variables - gender and response (disease status)

cbp1 <- c("#999999", "#E69F00", "#56B4E9", "#009E73",
          "#F0E442", "#0072B2", "#D55E00", "#CC79A7") #Color-blind friendly palatte

ggplot(data = train.data) + 
  geom_bar(mapping = aes(x = Category), fill = c(cbp1[3], cbp1[7])) + 
  labs(title = "Barplot of the Liver Disease Category", x = "Disease")


ggplot(data = train.data) + 
  geom_bar(mapping = aes(x = Gender), fill = c(cbp1[2], cbp1[4])) + 
  labs(title = "Barplot of the Genders of participants")
  

### Numerical variables

library(reshape2)

melt.train = melt(train.data[,-c(1, 3, 12)])

#Visualizing the distrubution of all the vairables
ggplot(data = melt.train, aes(x = value))+
  stat_density()+
  facet_wrap(~variable, scales = "free")

ggplot(data = melt.train, aes(x = value))+
  geom_histogram()+
  facet_wrap(~variable, scales = "free")

#Identifying the presence of outliers
boxplot(train.data)


# Visualizing variable relationships for insight into the data

### Correlation plots of all numerical variables

library(corrplot)
cors = cor(train.data[,-c(1, 3, 12)], use = "na.or.complete")

corrplot.mixed(cors)

### Distribution of disease category within genders

library(dplyr)

gen_cat_table = train.data %>% 
  group_by(Gender, Category) %>% 
  summarize(counts = n())

ggplot(gen_cat_table, aes(x = Category, y = counts))+
  geom_bar(position = "dodge", stat = "identity", aes(fill = Gender))


### Boxplot comparison of nuerical variables with Category




