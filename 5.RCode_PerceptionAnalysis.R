Sys.setlocale(locale = "Chinese")
library(dplyr)

# read the survey experiment data on incivility perceptions
perception_survey = readRDS("perception_survey.rds")

# test accuracy
adx = perception_survey %>% 
  group_by(Q) %>% 
  summarise(uncivil_dictionary=first(uncivil_dictionary), # dictionary based
            uncivil_perceptions=sum(uncivil_perceptions)/length(unique(ResponseId)), # perceptions from survey
            uncivil_coded=first(uncivil_coded)) # coded by human coders/experts

adx$uncivil_perceptions = ifelse(adx$uncivil_perceptions>0.5,1,0)

table(adx$uncivil_dictionary,adx$uncivil_perceptions) # (136+150)/500 = 0.572
#     0   1
# 0 136  23
# 1 191 150
table(adx$uncivil_perceptions,adx$uncivil_coded) # (124+156)/500 = 0.56
#     0   1
# 0 124 203
# 1  17 156
table(adx$uncivil_dictionary,adx$uncivil_coded) # (136+336)/500 = 0.944
#     0   1
# 0 136  23
# 1   5 336
#

perception_survey %>% 
  group_by(uncivil_dictionary) %>%
  summarise(m = mean(incivility))
# uncivil_dictionary     m
#                  0  3.52
#                  1  4.08
t.test(perception_survey$incivility~perception_survey$uncivil_dictionary)

# fixed effect model
library(lfe)
summary(mfix <- felm(incivility~uncivil_dictionary|ResponseId,perception_survey)) #0.5971 (se=0.0255) Adjusted R-squared: 0.6576 
