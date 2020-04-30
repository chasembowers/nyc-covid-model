import pandas as pd
import numpy as np
import math
from scipy.stats import lognorm
from scipy.optimize import fmin

# NYC COVID data
try:
    death_data = pd.read_csv(
        'https://raw.githubusercontent.com/nychealth/coronavirus-data/master/probable-confirmed-dod.csv',
        index_col=0,
        parse_dates=True)
    death_data.index = pd.DatetimeIndex(death_data.index)
    death_data = death_data.reindex(pd.date_range(death_data.index[0], death_data.index[-1]), fill_value=0)
    death_data.to_pickle('./deaths_backup')
except:
    death_data = pd.read_pickle('./deaths_backup')


FULL_DEATHS = death_data.replace(np.nan, 0)

# Remove the last week of data.
# https://www1.nyc.gov/site/doh/covid/covid-19-data.page?fbclid=IwAR21FOFuH6Doly_pbHfMzMlFs01zV57ljwNgWRODQVaOj8LQCDONyggVJpQ
# "Information about cases over the last week will be incomplete until the
# laboratories and hospitals report the results for people who were tested,
# which can take a few days to a week."
FULL_DEATHS = FULL_DEATHS.iloc[:-7]
LAST_DAY = FULL_DEATHS.index[-1]

DEATHS = FULL_DEATHS['CONFIRMED_COUNT'] + FULL_DEATHS['PROBABLE_COUNT']
DEATH_DAYS = len(DEATHS)

# This model hinges on the accuracy of fatality rate,
# which can be approximated in the general population, rather than the
# hospitalized population, for example, by testing everyone in a sample set
# in which severe cases are less overrepresented, like a flight or cruise ship.
# https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(20)30243-7/fulltext
FATAL_RATE = .0066

# Ratio of asymptomatic infected to total infected
# Estimate from the Diamond Princess is reasonable, given that
# "The reported proportion of infected individuals who were asymptomatic on the
# Diamond Princess did not vary considerably by age"
# https://www.cebm.net/covid-19/covid-19-what-proportion-are-asymptomatic/
# https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(20)30243-7/fulltext
ASYMPTOMATIC_RATIO = .18

# how far back from first day of dataset to simulate
#
# This is much earlier than first confirmed case in NYC because it takes about
# 45 days from the first infection for the growth of the
# contagious/infected populations to converge to exponential. This is
# unrealistic but prevents the model from overfitting in a high-dimensinal space
# of intital conditions. A fraction of a person can be infected to accomodate
# this.
DAYS_BEFORE_DEATHS = 45

# days before companies start instituting work from home
# Beginning of March was chosen by looking at Google Trends 'new york work from home'.
# https://trends.google.com/trends/explore?geo=US&q=new%20york%20work%20from%20home
DAYS_BEFORE_WFH = DAYS_BEFORE_DEATHS + DEATHS.index.get_loc('2020-03-11')+10

DAYS_BEFORE_PAUSE = DAYS_BEFORE_DEATHS + DEATHS.index.get_loc('2020-03-23')

# How long a person is contagious before/on/after symptom onset.
# vaguely modeled after these studies and influenza:
# https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(20)30196-1/fulltext
# https://www.nature.com/articles/s41591-020-0869-5
# https://www.cdc.gov/flu/about/disease/spread.htm
# "infectiousness profile may more closely resemble that of influenza than of SARS"
# "we inferred that infectiousness started from 2.3 days (95% CI, 0.8–3.0 days)
# before symptom onset and peaked at 0.7 days (95% CI, −0.2–2.0 days) before
# symptom onset (Fig. 1c). The estimated proportion of presymptomatic
# transmission (area under the curve) was 44% (95% CI, 25–69%). Infectiousness
# was estimated to decline quickly within 7 days."
# "People with flu are most contagious in the first 3-4 days after their illness begins."
DAYS_CONTAGIOUS_BEFORE = 3
DAYS_CONTAGIOUS_ON_AFTER = 5

# Incubation and death probabilties are binned by day, and days after these
# cut-offs are excluded.
MAX_INCUBATION = 14
MAX_ONSET_TO_DEATH = 30

# Incubation period assumed LogNormal(1.54, .47),
# as per https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(20)30230-9/fulltext
INC_PERIOD_DIST = lognorm(s=.47, scale=math.exp(1.54))
INC_PERIOD_PROB = {}
for day in range(MAX_INCUBATION+1):
    INC_PERIOD_PROB[day] = INC_PERIOD_DIST.cdf(day + .5) - INC_PERIOD_DIST.cdf(day - .5)

# onset to death periods assumed LogNormal(2.7, .45),
# which is a rough average of estimates in the following studies:
# https://www.nature.com/articles/s41591-020-0822-7
# https://www.mdpi.com/2077-0383/9/2/523/htm
# https://www.medrxiv.org/content/10.1101/2020.03.15.20036582v1.full.pdf
ONSET_TO_DEATH_DIST = lognorm(s=.45, scale=math.exp(2.7))
ONSET_TO_DEATH_PROB = {}
for day in range(MAX_ONSET_TO_DEATH+1):
    ONSET_TO_DEATH_PROB[day] = ONSET_TO_DEATH_DIST.cdf(day + .5) - ONSET_TO_DEATH_DIST.cdf(day - .5)

# NYC total population
TOTAL_POP = 8.4E6

# infection rate after the PAUSE
# From Governor Cuomo's daily briefings
R_AFTER_PAUSE = .75

# This model assumes presymptomatic transmission and ignores asymptomatic
# transmission because it seems much rarer.
# https://www.who.int/docs/default-source/coronaviruse/situation-reports/20200402-sitrep-73-covid-19.pdf?sfvrsn=5ae25bc7_2
#
# This model assumes constant R before March 1, a linear decrease in R from
# March 1 to March 22, and a new, constant R of .75 after March 22.
#
# X - array of (initial_contagious_log, R_before_wfh, R_before_pause) passed by fmin
# initial_contagious_log - Log of size of contagious population on day 1.
#   Not intended for realism but for scaling the infection curve.
#   Simulation will start with 1 day of contagiousness. Can be < 1.
# R_before_wfh- infection rate before March 1
# R_before_pause- infection rate on March 22.
#   Infection rate changes linearly from March 1 to March 22.
# score- if 'train', returns sum of squared errors of deaths predictions.
#   if 'test', returns accuracy of the last deaths prediction.
# days_excluded- exclude this many days from the end of the simulation.
#   used for fitting to a subset of available deaths data.
# days_predicted- add this many days to the end of the simulation.
# display- print statistics
def simulate(X, score='train', days_excluded=0, days_predicted=0, display=False):
    initial_contagious_log, R_before_wfh, R_before_pause = X

    if (R_before_wfh < 0 or
        R_before_pause < 0 or
        R_before_wfh < R_before_pause or
        R_before_pause < R_AFTER_PAUSE):
        return np.inf
    initial_contagious = math.exp(initial_contagious_log)
    if DEATH_DAYS - days_excluded <= DEATHS.index.get_loc('2020-03-23'):
        raise Exception('Must simulate on period that includes PAUSE.')
    if ((days_excluded != 0 and days_predicted != 0) or days_excluded < 0 or
        days_predicted < 0):
        raise Exception('Invalid days_excluded or days_predicted')

    days_simulated = DAYS_BEFORE_DEATHS + DEATH_DAYS - days_excluded + days_predicted
    scored_deaths = DEATHS.iloc[:len(DEATHS)-days_excluded]

    # Predicted number of contagious
    contagious = np.zeros(days_simulated)

    # Predicted number of newly symptomatic
    onset = np.zeros(days_simulated)

    # Predicted number of deaths
    pred_deaths = np.zeros(days_simulated)

    contagious[0] = initial_contagious

    # size of population contagious before or on day i
    curr_ever_contagious = initial_contagious

    for i in range(days_simulated-1):

        next_ever_contagious = curr_ever_contagious

        # fraction of population that has not been infected
        not_infected = (TOTAL_POP - float(curr_ever_contagious)/(1.-ASYMPTOMATIC_RATIO)) / TOTAL_POP

        # Calculate number of those infected on day i who will become contagious.
        # It is assumed that the infection rate is dampened proportionally to the fraction of NYC already infected.
        days_contagious = DAYS_CONTAGIOUS_BEFORE + DAYS_CONTAGIOUS_ON_AFTER
        if i < DAYS_BEFORE_WFH: become_contagious = contagious[i] * R_before_wfh / days_contagious * not_infected
        elif i < DAYS_BEFORE_PAUSE:
            x_intercept = (i + 1 - DAYS_BEFORE_WFH)/(DAYS_BEFORE_PAUSE - DAYS_BEFORE_WFH)
            become_contagious = contagious[i] * (R_before_wfh + x_intercept*(R_before_pause - R_before_wfh)) / days_contagious * not_infected
        else:
            become_contagious = contagious[i] * R_AFTER_PAUSE / days_contagious * not_infected

        # Increase contagious population of future days based on distribution
        # of incubation period and # of currently infected who become contagious.
        for inc_period in range(MAX_INCUBATION+1):

            # probability of onset in 'inc_period' days.
            prob = INC_PERIOD_PROB[inc_period]

            # expected number of infected with onset in 'inc_period' days
            become_contagious_today = become_contagious * prob

            # Increase contagious population for each day that infected
            # with onset in 'inc_period' days will be contagious.
            contagious[max(0,i+inc_period-DAYS_CONTAGIOUS_BEFORE):min(i+inc_period+DAYS_CONTAGIOUS_ON_AFTER, len(contagious))] += become_contagious_today

            # Increase population of ever contagious starting on the first day of cases.
            if i >= DAYS_BEFORE_DEATHS: next_ever_contagious += become_contagious_today

            # Increase number of newly symptomatic in 'inc_period' days.
            if i+inc_period<len(onset): onset[i+inc_period] += become_contagious_today

        curr_ever_contagious = next_ever_contagious

    # Predict deaths using array of # with symptom onset
    # and distribution of time between symptom onset and death.
    for i in range(len(onset)):
        for onset_to_death in range(MAX_ONSET_TO_DEATH+1):
            if i+onset_to_death<len(pred_deaths):

                # probability of death in 'onset_to_death' days.
                prob = ONSET_TO_DEATH_PROB[onset_to_death]

                # expected number dead in 'onset_to_death' days with symptom
                # on day i.
                died = onset[i] * FATAL_RATE / (1.-ASYMPTOMATIC_RATIO) * prob

                pred_deaths[i+onset_to_death] += died

    if display:
        print('Estimated confirmed + probable deaths: ', int(pred_deaths[-1]))
        infected = sum(onset) / (1. - ASYMPTOMATIC_RATIO)
        print('Estimated ever infected: {} ({:.2f}%)'.format(int(infected), 100*float(infected)/TOTAL_POP))
        print('Estimated currently contagious: {} ({:.2f}%)'.format(int(contagious[-1]), 100*contagious[-1]/TOTAL_POP))
        print('Estimated maximum ever simultaneously contagious: {} ({:.2f}%)'.format(int(max(contagious)), 100*max(contagious)/TOTAL_POP))

    # Cost function for training is Sum of Squares.
    if score == 'train':
        rsquared_death = 1.-sum((scored_deaths - pred_deaths[-len(scored_deaths):])**2) / sum((scored_deaths-np.mean(scored_deaths))**2)
        return -rsquared_death

    # Training score is accuracy of last deaths prediction.
    if score == 'test':

        deaths_actual = scored_deaths.iloc[-1]
        deaths_pred = pred_deaths[-1]

        return 1.-abs(deaths_pred-deaths_actual)/float(deaths_actual)

# Fit the model, excluding the last 'days_excluded' days from the training set.
def fitted_parameters(days_excluded):
    return fmin(simulate, [3.27080831, 4.10536956, 0.73740359],
                args=('train',days_excluded,0,False),
                disp=False)

print('Training/predicting on limited subsets of death data to estimate prediction accuracy...\n')
death_scores = []
for i in range(7):
    params = fitted_parameters(days_excluded=i+8)
    scores = simulate(params, score='test', days_excluded=i, display=False,)
    death_scores.append(scores)
print('Mean accuracy 8-days-out deaths prediction over last 7 known days: {:.2f}% \n'.format(100*np.mean(death_scores)))
print('Training on all of death data...\n')
params = fitted_parameters(days_excluded=0)
print('Prediction in NYC for {} :\n'.format((LAST_DAY + pd.DateOffset(8)).strftime('%Y-%m-%d')))
simulate(params, score='test', days_excluded=0, days_predicted=7, display=True)
