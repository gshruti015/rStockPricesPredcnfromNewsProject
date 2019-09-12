#install.packages("e1071")
#install.packages("SparseM")
#install.packages("tm")
library('e1071')
library('SparseM')
library('tm')
library('plotly')
library('ggplot2')
library('ggthemes')
library('wordcloud')
library("RColorBrewer")
library('readr')

setwd("~/Desktop/Sem2/Data MIning - Kislay Prasad/Data Mining Project")
news <- read.csv("RedditNews.csv", stringsAsFactors = FALSE)
#with(df, aggregate(list(news$News = news$News), list(news$Date = news$Date), sum))
news$News <- as.character(news$News)
newsvector <- as.vector(news$News);    # Create vector
newssource <- VectorSource(newsvector); # Create source
newscorpus <- Corpus(newssource);       # Create corpus

#################################################################################
# PERFORMING THE VARIOUS TRANSFORMATIONS on "traincorpus" and "testcorpus" DATASETS 
# SUCH AS TRIM WHITESPACE, REMOVE PUNCTUATION, REMOVE STOPWORDS.
newscorpus <- tm_map(newscorpus,stripWhitespace);
newscorpus <- tm_map(newscorpus,content_transformer(tolower));
newscorpus <- tm_map(newscorpus, removeWords,c("the","end",stopwords("english")));
newscorpus <- tm_map(newscorpus,removePunctuation);
newscorpus <- tm_map(newscorpus,removeNumbers);
inspect(newscorpus[1])
#newscorpus 
# remove anything other than English letters or space
removeNumPunct <- function(x) gsub("[^[:alpha:][:space:]]*", "", x)
newscorpus <- tm_map(newscorpus, content_transformer(removeNumPunct))

# remove URLs
removeURL <- function(x) gsub("http[[:alnum:]]*", "", x)
newscorpus <- tm_map(newscorpus, content_transformer(removeURL))

# Create TermDocumentMatrix
tdm1 <- TermDocumentMatrix(newscorpus)
tdm1 = removeSparseTerms(tdm1, 0.99)
tdm1
#tdm2->data.frame(tdm1)
## CREATE DOCUMENT TERM MATRIX
dtm_matrix <- t(tdm1)
dtm_matrix
dtm_df <- data.frame(news$Date,as.matrix(dtm_matrix))
dtm_df

#combined news from all dates
combined_news<-aggregate(dtm_df[-1], by=list(dtm_df$news.Date), sum)

#news from stock data
df1<-read.csv("DJIA_table.csv")
combined_news$Date<-combined_news$Group.1
total <- merge(combined_news,df1,by="Date")
total$Group.1<-NULL

#viz on combined news
viz_data<-combined_news
viz_data<-viz_data[-1]
viz_data<-viz_data[-114]
freq_df<-data.frame(names(viz_data),colSums(viz_data))

#ordered term-frequency bar plot
freq_df <- freq_df[order(-freq_df$colSums.viz_data.),] 
p <- plot_ly(
  x = freq_df$names.viz_data.,
  y = freq_df$colSums.viz_data.,
  name = "Bar Plot of Term Frequency",
  type = "bar", text = "Primates", textposition = 'middle right', textfont = list(color = '#000000', size = 16)) %>%
  layout(title = "Bar Plot of Term Frequency",
         xaxis = list(title = 'Keywords from Twitter Data',
                      zeroline = TRUE,
         yaxis = list(title = 'Frequency of each Kewyord', zeroline = TRUE)))

findFreqTerms(freq_df, lowfreq = 4)

#word cloud of important words
library('wordcloud')
wordcloud(words = freq_df$names.viz_data., freq = freq_df$colSums.viz_data., min.freq = 1,
          max.words=1000, random.order=FALSE, rot.per=0.35, scale=c(4,.5),
          colors=brewer.pal(12, "Paired"))

library(wordcloud2)
wordcloud2(data = freq_df)


#Boosting
set.seed(123)
inTrain <- sample(nrow(total),0.7*nrow(total))
traindata <- total[inTrain,]
testdata <- total[-inTrain,]
final<-total
final<-final[-1]

#copying train and test data for plotting with date
traindata1<-traindata[-2]
testdata1<-testdata[-2]

library(gbm)
set.seed(1)
boost.Close=gbm(Close~.-Adj.Close-High-Low-Volume,data=final[inTrain,],n.trees=5000,interaction.depth=4)
summary(boost.Close)
par(mfrow=c(1,2))
yhat.boost=predict(boost.Close,newdata=final[-inTrain,],n.trees=5000,type="response")

test_Val<-testdata1$Close
comp<-data.frame(testdata1$Date,test_Val,yhat.boost)
colnames(comp)<-c("Dates", "Test_Values", "Predicted_Values")


library(lubridate)
comp$year<-year(as.Date(comp$Dates))
comp$weekday<-weekdays(as.Date(comp$Dates))
comp$day<-day(as.Date(comp$Dates))
comp$week<-week(as.Date(comp$Dates))
comp$weekyear<-paste0(comp$week,",",comp$year)

comp_table <- comp %>% 
  select(Test_Values, Predicted_Values) %>%
  group_by(comp$weekyear) %>%
  summarise(Test_Values = mean(comp$Test_Values), Predicted_Values = mean(comp$Predicted_Values))

colnames(comp_table)<-c("WeekYear", "Test_Values", "Predicted_Values")

#decide the best 
q <- plot_ly(data = comp, x = comp$weekyear, y = comp$Test_Values, type = 'scatter',
             marker = list(size = 10,
                           color = 'rgba(255, 182, 193, .9)',
                           line = list(color = 'rgba(152, 0, 0, .8)',
                                       width = 2))) %>%
  layout(title = 'Styled Scatter',
         yaxis = list(zeroline = FALSE),
         xaxis = list(zeroline = FALSE))

q <- plot_ly(data = comp, x = comp_table$WeekYear, y = comp_table$Test_Values, type = 'scatter',
             marker = list(size = 10,
                           color = 'rgba(255, 182, 193, .9)',
                           line = list(color = 'rgba(152, 0, 0, .8)',
                                       width = 2))) %>%
  layout(title = 'Styled Scatter',
         yaxis = list(zeroline = FALSE),
         xaxis = list(zeroline = FALSE))

#analyzing text data sentiments
library(tidytext)
library(dplyr)
news$Date<-as.POSIXct(news$Date)
tweets<- news %>%
  select(Date,News) %>%
  unnest_tokens_("word", text)


