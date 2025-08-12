library(WCE)

# Create a drugdata dataset for testing
drugdata <- WCE::drugdata


# Save the dataset to a csv file
write.csv(drugdata, file = "drugdata.csv", row.names = FALSE)