# nyc-covid-model
  Train/test a model of the COVID outbreak in New York City accounting for the PAUSE (lockdown). The model is fit to the publicly available death data using the Lancet's fatality rate, Governor Cuomo's current infection rate, and statistics from a bunch of other studies. The model predicts future deaths and the size of infected/contagious populations.  

    Training/predicting on limited subsets of death data to estimate prediction accuracy...

    Mean accuracy 8-days-out deaths prediction over last 7 known days: 92.52% 

    Training on all of death data...

    Prediction in NYC for 2020-04-28 :

    Estimated confirmed + probable deaths:  197
    Estimated ever infected: 3013352 (35.87%)
    Estimated currently contagious: 31128 (0.37%)
    Estimated maximum ever simultaneously contagious: 1088032 (12.95%)

## Testing Accuracy

The number of confirmed and probable deaths are "predicted back" for each of the last 7 days with complete data, with the model ignorant of each predicted day and 7 days previous to it. As of April 28, the average prediction accuracy is 92.52%. No (hyper)parameters were tuned to these last 7 days, but some model choices were influenced by them, so accuracy in practice is expected to be lower.

## The Model

The model assumes that the infection rate R was constant before March 1, about when companies started instituting work from home policies. Then there is a linear decrease in R from the previous value on March 1 to some other value on the day before the first full day of the PAUSE (lockdown), March 23. Then R=.75 after March 23, as per Governor Cuomo's briefing. Everything is predicted by stepping through each day in simulation. Each day, expected numbers of contagious people are added to future days based on the conditions of that day. Probability distributions, rather than constants, are used for the incubation period and the time between symptom onset and death. The model uses an infection-fatality rate from The Lancet of .66%.
