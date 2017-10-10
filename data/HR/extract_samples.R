clean_data <- function(data) {
    data <- execute_replace_categorical(data)
    data <- remove_categorical(data)
    data <- data.frame(apply(data, 2, replace_na_with_median_value))
    # data <- data[sample(nrow(data), replace=F),]
}

clean_data_with_one_hot <- function(data, one_hot_var) {
    factor_variables <- which(sapply(data[1,],class)=="factor")
    data_factor <- data[, factor_variables]
    data_factor_one_hot <- dummy.data.frame(data_factor[,one_hot_var], sep="_")

    to_remove <- NULL

    for (i in 1:ncol(data_factor_one_hot)){
        if(sum(data_factor_one_hot[,i]) == 0){
            to_remove <- cbind(to_remove, i)
        }
    }
    # print(to_remove)

    data_factor_one_hot <- data_factor_one_hot[,setdiff(1:ncol(data_factor_one_hot), to_remove)] 

    new_data <- execute_replace_categorical(data)
    new_data <- remove_categorical(new_data)
    new_data <- data.frame(apply(new_data, 2, replace_na_with_median_value))
    final_data <- cbind(new_data, data_factor_one_hot)
    # final_data <- final_data[sample(nrow(final_data), replace=F),]
}


# Replace the categorical variable column having factored values in "values_name" 
# by the corresponding integer value in "values_int"
replace_categorical <- function(data, column, values_names, values_int){
    for (i in 1:length(values_names)){
    # for (i in 1:1){
        data[[column]] <- as.character(data[[column]])
        indices <- which(data[,column] == values_names[i])
        data[indices, column] <- values_int[i]
    }
    data[[column]] <- as.integer(data[[column]])
    new_data <- data
    new_data
}

execute_replace_categorical <- function(data){
    data <- replace_categorical(data, "LandSlope", c('Gtl', 'Mod', 'Sev'), c("3", "2", "1"))
    data <- replace_categorical(data, "ExterQual", c('Ex', 'Gd', 'TA', 'Fa', 'Po'), c("5", "4", "3", "2", "1"))
    data <- replace_categorical(data, "ExterCond", c('Ex', 'Gd', 'TA', 'Fa', 'Po'), c("5", "4", "3", "2", "1"))

    data <- replace_categorical(data, "BsmtQual", c('Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'), c("6", "5", "4", "3", "2", "1"))
    data <- replace_categorical(data, "BsmtCond", c('Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'), c("6", "5", "4", "3", "2", "1"))
    data <- replace_categorical(data, "BsmtExposure", c('Gd', 'Av', 'Mn', 'No', 'NA'), c("5", "4", "3", "2", "1"))
    data <- replace_categorical(data, "BsmtFinType1", c('GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA'),
                                c("7", "6", "5", "4", "3", "2", "1"))
    data <- replace_categorical(data, "BsmtFinType2", c('GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA'),
                                c("7", "6", "5", "4", "3", "2", "1"))

    data <- replace_categorical(data, "CentralAir", c('Y', 'N'), c("1", "0"))
    data <- replace_categorical(data, "HeatingQC", c('Ex', 'Gd', 'TA', 'Fa', 'Po'), c("5", "4", "3", "2", "1"))

    data <- replace_categorical(data, "KitchenQual", c('Ex', 'Gd', 'TA', 'Fa', 'Po'), c("5", "4", "3", "2", "1"))

    data <- replace_categorical(data, "FireplaceQu", c('Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'), c("6", "5", "4", "3", "2", "1"))


    data <- replace_categorical(data, "GarageFinish", c('Fin', 'RFn', 'Unf', 'NA'), c("4", "3", "2", "1"))
    data <- replace_categorical(data, "GarageQual", c('Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'), c("6", "5", "4", "3", "2", "1"))
    data <- replace_categorical(data, "GarageCond", c('Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'), c("6", "5", "4", "3", "2", "1"))

    data <- replace_categorical(data, "PoolQC", c('Ex', 'Gd', 'TA', 'Fa', 'NA'), c("5", "4", "3", "2", "1"))
    data
}

# Remove all categorical variables and Id column
remove_categorical <- function(data){
    factor_variables<-which(sapply(data[1,],class)=="factor")
    data_preprocessed<-data[,-factor_variables]
    data_preprocessed <- data_preprocessed[,2:length(data_preprocessed[1,])] # remove Id 
    data_preprocessed
}

# Two functions that replace the missing values with the mean or median of the evaluations
replace_na_with_mean_value<-function(vec) {
    mean_vec<-mean(vec,na.rm=T)
    vec[is.na(vec)]<-mean_vec
    vec
}
replace_na_with_median_value<-function(vec) {
    median_vec<-median(vec,na.rm=T)
    vec[is.na(vec)]<-median_vec
    vec
}

data<-read.csv("kaggle_form.csv")
data <- replace_categorical(data, "salary", c('high', 'medium', 'low'), c("3", "2", "1"))
factor_variables<-which(sapply(data[1,],class)=="factor")
data_preprocessed<-data[,-factor_variables]
data <- data_preprocessed
data[,6] <- - data[,6]
data[,7] <- - data[,7]
data <-data[,-2]
summary(data)
# write.csv(data, 'raw', row.names=F)
