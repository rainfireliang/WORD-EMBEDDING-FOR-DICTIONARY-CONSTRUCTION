Sys.setlocale(locale = "Chinese")
library(ggplot2)
library(jtools)
library(readr)
library(tidyverse)
library(purrrlyr)
library(zoo)
library(eoffice)

## Fig2A ##

# read vulgarity words collected in each enrichment round (manually filtered)
vulgarity = readRDS('vulgarity_byRound.rds') 
agg_vulgarity = vulgarity %>% group_by(Round) %>% summarise(n=length(Words))
agg_vulgarity$incivility_type = "Vulgarity"
# read name calling words collected in each enrichment round (manually filtered)
ncall = readRDS('namecalling_byROund.rds') 
agg_ncall = ncall %>% group_by(Round) %>% summarise(n=length(Words))
agg_ncall$incivility_type = "Name-calling"
# combined for plot
ByRounds = bind_rows(agg_vulgarity,agg_ncall)
# plot
fg2A = ggplot(data=ByRounds,aes(x=Round, y=n, linetype=incivility_type)) + geom_line() + ylab("Number of new words") +
  scale_x_continuous(breaks=c(1:16))+
  theme_apa(legend.pos = "topright")

## Fig2B ##
# random seeds = 10
data10 <- read_delim("coverage_10.txt",delim = "\t", escape_double = FALSE,col_names = FALSE)
colnames(data10) <- c('Rep','Round','total','Coverage#','Coverage%')
data10$seed <- 10

data = data10 %>% select('Rep','Round','Coverage#') %>% pivot_wider(names_from = 'Round', values_from = 'Coverage#')
data = data %>% pivot_longer(cols = c(2:19),names_to = 'Round', values_to = 'Coverage#')
data = data %>% 
  slice_rows("Rep") %>% 
  by_slice(function(x) { 
    na.locf(na.locf(x, na.rm=F), fromLast=T, na.rm=F) },
    .collate = "rows") 
data$Round = as.numeric(data$Round)
data10 <- data %>% group_by(Round) %>% summarise(mdn=median(`Coverage#`),
                                                 min=min(`Coverage#`),
                                                 max=max(`Coverage#`))
# random seeds = 50
data50 <- read_delim("coverage_50.txt",delim = "\t", escape_double = FALSE,col_names = FALSE)
colnames(data50) <- c('Rep','Round','total','Coverage#','Coverage#')
data50$seed <- 50

data = data50 %>% select('Rep','Round','Coverage#') %>% pivot_wider(names_from = 'Round', values_from = 'Coverage#')
data = data %>% pivot_longer(cols = c(2:15),names_to = 'Round', values_to = 'Coverage#')
data = data %>% 
  slice_rows("Rep") %>% 
  by_slice(function(x) { 
    na.locf(na.locf(x, na.rm=F), fromLast=T, na.rm=F) },
    .collate = "rows") 
data$Round = as.numeric(data$Round)
data50 <- data %>% group_by(Round) %>% summarise(mdn=median(`Coverage#`),
                                                 min=min(`Coverage#`),
                                                 max=max(`Coverage#`))
# random seeds = 100
data100 <- read_delim("coverage_100.txt",delim = "\t", escape_double = FALSE,col_names = FALSE)
colnames(data100) <- c('Rep','Round','total','Coverage#','Coverage#')
data100$seed <- 100

data = data100 %>% select('Rep','Round','Coverage#') %>% pivot_wider(names_from = 'Round', values_from = 'Coverage#')
data = data %>% pivot_longer(cols = c(2:15),names_to = 'Round', values_to = 'Coverage#')
data = data %>% 
  slice_rows("Rep") %>% 
  by_slice(function(x) { 
    na.locf(na.locf(x, na.rm=F), fromLast=T, na.rm=F) },
    .collate = "rows") 
data$Round = as.numeric(data$Round)
data100 <- data %>% group_by(Round) %>% summarise(mdn=median(`Coverage#`),
                                                  min=min(`Coverage#`),
                                                  max=max(`Coverage#`))
# combined for plot
data10$seed = 10
data50$seed = 50
data100$seed = 100
data = bind_rows(data10,data50,data100)

# plot
fg2B <- ggplot(data, aes(x=Round, y=mdn, group=seed, linetype=factor(seed))) + 
  geom_hline(yintercept = 1911,lty = 2) +
  geom_line() +
  scale_x_continuous(breaks = c(1:18)) +
  scale_y_continuous(breaks = seq(0, 1800, by = 200)) +
  theme_apa()+
  labs(x="Round", y = "Number of uncivil words", color = "Number of seed words", shape = "Number of seed words") +
  theme(legend.position = c(0.8, 0.4)) 

## Fig2C ##
# random seeds = 10
data10 <- read_delim("coverage_10.txt",delim = "\t", escape_double = FALSE,col_names = FALSE)
colnames(data10) <- c('Rep','Round','total','Coverage#','Coverage%')
data10$seed <- 10

data = data10 %>% select('Rep','Round','Coverage%') %>% pivot_wider(names_from = 'Round', values_from = 'Coverage%')
data = data %>% pivot_longer(cols = c(2:19),names_to = 'Round', values_to = 'Coverage%')
data = data %>% 
  slice_rows("Rep") %>% 
  by_slice(function(x) { 
    na.locf(na.locf(x, na.rm=F), fromLast=T, na.rm=F) },
    .collate = "rows") 
data$Round = as.numeric(data$Round)
data10 <- data %>% group_by(Round) %>% summarise(mdn=median(`Coverage%`),
                                                 min=min(`Coverage%`),
                                                 max=max(`Coverage%`))
# random seeds = 50
data50 <- read_delim("coverage_50.txt",delim = "\t", escape_double = FALSE,col_names = FALSE)
colnames(data50) <- c('Rep','Round','total','Coverage#','Coverage%')
data50$seed <- 50

data = data50 %>% select('Rep','Round','Coverage%') %>% pivot_wider(names_from = 'Round', values_from = 'Coverage%')
data = data %>% pivot_longer(cols = c(2:15),names_to = 'Round', values_to = 'Coverage%')
data = data %>% 
  slice_rows("Rep") %>% 
  by_slice(function(x) { 
    na.locf(na.locf(x, na.rm=F), fromLast=T, na.rm=F) },
    .collate = "rows") 
data$Round = as.numeric(data$Round)
data50 <- data %>% group_by(Round) %>% summarise(mdn=median(`Coverage%`),
                                                 min=min(`Coverage%`),
                                                 max=max(`Coverage%`))
# random seeds = 100
data100 <- read_delim("coverage_100.txt",delim = "\t", escape_double = FALSE,col_names = FALSE)
colnames(data100) <- c('Rep','Round','total','Coverage#','Coverage%')
data100$seed <- 100

data = data100 %>% select('Rep','Round','Coverage%') %>% pivot_wider(names_from = 'Round', values_from = 'Coverage%')
data = data %>% pivot_longer(cols = c(2:15),names_to = 'Round', values_to = 'Coverage%')
data = data %>% 
  slice_rows("Rep") %>% 
  by_slice(function(x) { 
    na.locf(na.locf(x, na.rm=F), fromLast=T, na.rm=F) },
    .collate = "rows") 
data$Round = as.numeric(data$Round)
data100 <- data %>% group_by(Round) %>% summarise(mdn=median(`Coverage%`),
                                                  min=min(`Coverage%`),
                                                  max=max(`Coverage%`))
# combined for plot
data10$seed = 10
data50$seed = 50
data100$seed = 100
data = bind_rows(data10,data50,data100)
# plot
fg2C <- ggplot(data, aes(x=Round, y=mdn, group=seed, linetype=factor(seed))) + 
  geom_hline(yintercept = 0.9574114069984,lty = 2) +
  geom_line() +
  scale_x_continuous(breaks = c(1:18)) +
  scale_y_continuous(breaks = seq(0.00, 1.00, by = 0.10)) +
  theme_apa()+
  labs(x="Round", y = "Coverage", color = "Number of seed words", shape = "Number of seed words") +
  theme(legend.position = c(0.8, 0.6)) 


## combine fg2A/B/C into one ##
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  require(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}
fig2 = multiplot(fg2A,fg2B,fg2C,cols = 1)
topptx(fig2,"Fig2.pptx")
