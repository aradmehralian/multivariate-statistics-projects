library(psych)
library(GPArotation)
library(lavaan)

load("Data/fsdata.Rdata")
sum(is.na(fsdata)) # checking for missing values

number_columns <- ncol(fsdata)
print(number_columns)

## part a

# scaling data and dropping the "country" column
df_standard <- as.data.frame(scale(fsdata[, 2:number_columns],
                                   center=T, scale=T))


# calculating the correlation matrix
cor_df <- cor(df_standard)

# conducting EFA
efa_df <- fa(cor_df, fm="mle", nfactors=5, rotate="oblimin",
             scores="regression")
print(efa_df)


## part b

# centering data
df_centered <- as.data.frame(scale(fsdata[, 2:number_columns],
                                   center=T, scale=F))

# calculating the covariance matrix
cov_df <- cov(df_centered)
sample_size <- nrow(df_centered)
print(sample_size)

# model parameters
cfa_model_1 <- '
FS=~NA*FS_pay_bills+FS_afford_extras+FS_afford_housing+FS_save_money
FSF=~NA*FSF_pay_bills+FSF_afford_extras+FSF_afford_housing+FSF_save_money
SFJ=~NA*SFJ_no_info+SFJ_no_chance_show+SFJ_no_training+SFJ_no_support_findjob
SDJ=~NA*SDJ_help_people+SDJ_learn_new_things+SDJ_develop_creativity+
SDJ_meet_people+SDJ_feeling_self_worth
HEALTH=~NA*HEALTH_felt_down+HEALTH_limitation
FS ~~ 1*FS
FSF ~~ 1*FSF
SFJ ~~ 1*SFJ
SDJ ~~ 1*SDJ
HEALTH ~~ 1*HEALTH
FS ~~ FSF
FS ~~ SFJ
FS ~~ SDJ
FS ~~ HEALTH
FSF ~~ SFJ
FSF ~~ SDJ
FSF ~~ HEALTH
SFJ ~~ SDJ
SFJ ~~ HEALTH
SDJ ~~ HEALTH
'

# fitting the data and examining the result
cfa_fit_1 <- cfa(model=cfa_model_1, sample.cov=cov_df, sample.nobs=sample_size)
summary(cfa_fit_1, fit.measures=T)
# standardizedsolution(cfa_fit_1)

## part c

# getting the modification indices
mi_fit_1 <- modificationindices(cfa_fit_1)

# sorting based on largest drop in Chi score
mi_order <- order(mi_fit_1$mi, decreasing=T)
mi_fit1_sorted <- mi_fit_1[mi_order,]

# constraining 8 parameters to improve performance
cfa_model_2 <- '
FS=~NA*FS_pay_bills+FS_afford_extras+FS_afford_housing+FS_save_money
FSF=~NA*FSF_pay_bills+FSF_afford_extras+FSF_afford_housing+FSF_save_money
SFJ=~NA*SFJ_no_info+SFJ_no_chance_show+SFJ_no_training+SFJ_no_support_findjob
SDJ=~NA*SDJ_help_people+SDJ_learn_new_things+SDJ_develop_creativity+
SDJ_meet_people+SDJ_feeling_self_worth
HEALTH=~NA*HEALTH_felt_down+HEALTH_limitation
FS ~~ 1*FS
FSF ~~ 1*FSF
SFJ ~~ 1*SFJ
SDJ ~~ 1*SDJ
HEALTH ~~ 1*HEALTH
FS ~~ FSF
FS ~~ SFJ
FS ~~ SDJ
FS ~~ HEALTH
FSF ~~ SFJ
FSF ~~ SDJ
FSF ~~ HEALTH
SFJ ~~ SDJ
SFJ ~~ HEALTH
SDJ ~~ HEALTH
FSF_afford_extras ~~ FSF_save_money
SDJ_help_people ~~ SDJ_meet_people
SDJ_learn_new_things ~~ SDJ_develop_creativity
FS_pay_bills ~~ FSF_pay_bills
FS_afford_housing ~~ FSF_afford_housing
FS_save_money ~~ FSF_save_money
FS_afford_extras ~~ FSF_afford_extras
FS_save_money ~~ FS_afford_extras
'

cfa_fit_2 <- cfa(model=cfa_model_2, sample.cov=cov_df, sample.nobs=sample_size)
summary(cfa_fit_2, fit.measures=T)

## part d
df_centered$country <- fsdata$country

model_structural_free <- '
FS=~NA*FS_pay_bills+FS_afford_extras+FS_afford_housing+FS_save_money
FSF=~NA*FSF_pay_bills+FSF_afford_extras+FSF_afford_housing+FSF_save_money
SFJ=~NA*SFJ_no_info+SFJ_no_chance_show+SFJ_no_training+SFJ_no_support_findjob
SDJ=~NA*SDJ_help_people+SDJ_learn_new_things+SDJ_develop_creativity+
SDJ_meet_people+SDJ_feeling_self_worth
HEALTH=~NA*HEALTH_felt_down+HEALTH_limitation

FS ~~ 1*FS
FSF ~~ 1*FSF
SFJ ~~ 1*SFJ
SDJ ~~ 1*SDJ
HEALTH ~~ 1*HEALTH
FS ~~ FSF
FS ~~ SFJ
FS ~~ SDJ
FS ~~ HEALTH
FSF ~~ SFJ
FSF ~~ SDJ
FSF ~~ HEALTH
SFJ ~~ SDJ
SFJ ~~ HEALTH
SDJ ~~ HEALTH

FSF_afford_extras ~~ FSF_save_money
SDJ_help_people ~~ SDJ_meet_people
SDJ_learn_new_things ~~ SDJ_develop_creativity
FS_pay_bills ~~ FSF_pay_bills
FS_afford_housing ~~ FSF_afford_housing
FS_save_money ~~ FSF_save_money
FS_afford_extras ~~ FSF_afford_extras
FS_save_money ~~ FS_afford_extras

FS ~ FSF + SFJ + SDJ + HEALTH
'

model_structural_constrained <- '
FS=~NA*FS_pay_bills+FS_afford_extras+FS_afford_housing+FS_save_money
FSF=~NA*FSF_pay_bills+FSF_afford_extras+FSF_afford_housing+FSF_save_money
SFJ=~NA*SFJ_no_info+SFJ_no_chance_show+SFJ_no_training+SFJ_no_support_findjob
SDJ=~NA*SDJ_help_people+SDJ_learn_new_things+SDJ_develop_creativity+
SDJ_meet_people+SDJ_feeling_self_worth
HEALTH=~NA*HEALTH_felt_down+HEALTH_limitation

FS ~~ 1*FS
FSF ~~ 1*FSF
SFJ ~~ 1*SFJ
SDJ ~~ 1*SDJ
HEALTH ~~ 1*HEALTH
FS ~~ FSF
FS ~~ SFJ
FS ~~ SDJ
FS ~~ HEALTH
FSF ~~ SFJ
FSF ~~ SDJ
FSF ~~ HEALTH
SFJ ~~ SDJ
SFJ ~~ HEALTH
SDJ ~~ HEALTH

FSF_afford_extras ~~ FSF_save_money
SDJ_help_people ~~ SDJ_meet_people
SDJ_learn_new_things ~~ SDJ_develop_creativity
FS_pay_bills ~~ FSF_pay_bills
FS_afford_housing ~~ FSF_afford_housing
FS_save_money ~~ FSF_save_money
FS_afford_extras ~~ FSF_afford_extras
FS_save_money ~~ FS_afford_extras

FS ~ a1*FSF + a2*SFJ + a3*SDJ + a4*HEALTH
'

# d.1
config_invariance <- sem(model_structural_free, data=df_centered,
                         group="country")

summary(config_invariance, fit.measures=T)


# d.2
config_invariance_equal <- sem(model_structural_constrained, data=df_centered,
                               group="country", group.equal="regressions")

summary(config_invariance_equal, fit.measures=T)


# d.3
metric_invariance <- sem(model_structural_free, data=df_centered, 
            group="country", group.equal="loadings")

summary(metric_invariance)


# d.4
metric_invariance_equal <- sem(model_structural_constrained, data=df_centered, 
                               group="country",
                               group.equal=c("loadings", "regressions"))

summary(metric_invariance_equal)

# comparing fit measurements
fitmeasures(config_invariance, c("chisq","df","cfi","tli","rmsea","srmr"))
fitmeasures(config_invariance_equal, c("chisq","df","cfi","tli","rmsea","srmr"))
fitmeasures(metric_invariance, c("chisq","df","cfi","tli","rmsea","srmr"))
fitmeasures(metric_invariance_equal, c("chisq","df","cfi","tli","rmsea","srmr"))
