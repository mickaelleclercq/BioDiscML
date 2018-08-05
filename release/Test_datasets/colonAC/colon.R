load("E:/cloud/Code/BruteForceML/benchmark/colonCA/colon.rda")
colon_data=cbind(colon.x,colon.y)
colnames(colon_data) = paste0("V", 1:ncol(colon_data))
colon=as.data.frame(colon_data)
dim(colon)
colon$V2001
brain$V2001 = gsub(0,"A",brain$V2001)
brain$V2001 = gsub(1,"B",brain$V2001)
write.table(colon,file="colon.csv",sep="\t", row.names=T, col.names=T, quote=FALSE)