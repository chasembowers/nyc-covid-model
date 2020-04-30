# nyc-covid-model
  Train/test a model of the COVID outbreak in New York City accounting for the PAUSE (lockdown). The model is fit to the publicly available death data using the Lancet's fatality rate, Governor Cuomo's current infection rate, and statistics from a bunch of other studies. The model predicts future deaths and the size of infected/contagious populations.  


    Training/predicting on limited subsets of death data to estimate prediction accuracy...

    Mean accuracy 8-days-out deaths prediction over last 14 known days: 85.12% 

    Training on all of death data...

    Prediction in NYC for 2020-04-28 :

    Estimated confirmed + probable deaths:  191
    Estimated ever infected: 3116535 (37.10%)
    Estimated currently contagious: 29873 (0.36%)
    Estimated maximum ever simultaneously contagious: 1051559 (12.52%)


![alt tag](https://github.com/chasembowers/nyc-covid-model/raw/master/nyc_covid.png)

## Testing Accuracy

The number of confirmed and probable deaths are "predicted back" for each of the last N days with complete data, with the model ignorant of each predicted day and 7 days previous to it. The last 7 days with complete data were conserved for out-of-sample testing, but out-of-sample accuracy was higher than in-sample. As of April 28, the average prediction accuracy for the last 14 days is 85.12%.

## The Model

The model assumes that the infection rate R was constant before March 23, then equal to .75 afterward, as per Governor Cuomo's briefing. Everything is predicted by stepping through each day in simulation. Each day, expected numbers of contagious people are added to future days based on the conditions of that day. Probability distributions, rather than constants, are used for the incubation period and the time between symptom onset and death. The model uses an infection-fatality rate from The Lancet of .66%.
