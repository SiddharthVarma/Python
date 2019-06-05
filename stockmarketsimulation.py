import random

import numpy as np

import statistics

import scipy.stats

from matplotlib import pyplot as plt

#######SETS INITIAL VARIABLES#############-------------------------------------

mu = 0.000090 #drift (higher drift, more exponential the graph becomes)

sigma = 0.01 #volatility (hwo much the share price changes in each step)

deltas = 19

days = 1999 #NO. OF DAYS TO FIND THE SHARE PRICES TO

iteration = int(100)

###############################################################################

#CREATES time axis --------------------------------------------------
t = [0]

for l in range(days):
    t.append(l)
    
finalshareprice = []

for i in range(iteration):

    sprice = [252.0]            #SETS ARRAY WITH AN INITIAL PRICE OF 252 DOLLARS

    for x in range(days):


        deltas = mu*sprice[x]+sigma*random.gauss(0.0,1.0)*sprice[x]*random.gauss(0.0,1.0) #creating GBM model equation from script
  
        sprice.append(sprice[x]+deltas)


    finalshareprice.append(sprice[-1]) #-1 takes the last value of sprice and puts it into a new array called finalshareprice

    plt.plot(t, sprice)
    plt.title('Python Model of share price')
    plt.xlabel('number of days')
    plt.ylabel('Share Price ($)')
###########################################################################################################

plt.savefig('allofshareprice.png', bbox_inches='tight')
plt.show()

##FINDS MIN, MAX, VARIANCE, MEDIAN, MEAN, AND SORTS THE VALUES OF FINAL SHARE PRICE FROM LOW TO HIGH#######
sortprice = np.sort(sprice) ####sorts the sprice low to high

#print (sortprice)

print ('-------------------------------------------------------')

print ('-------------------------------------------------------')

finalsort = np.sort(finalshareprice)

finalshareprice = np.asarray(finalshareprice)

med = np.median(finalshareprice)

minimum = finalshareprice.min()

maximum = finalshareprice.max()

mean = finalshareprice.mean()

var = statistics.variance(finalshareprice, mean)

print ('the median value is', med)

print ('the min value is', minimum)

print ('the max value is', maximum)

print ('the mean value is', mean)

print ('the variance is', var)

sd = np.std(finalsort)
error = sd/(iteration**0.5) #finds error in finalshareprice


print ('standard deviation', sd)

print ('standard error is ', error)

print ('-------------------------------------------------------')

print ('-------------------------------------------------------')

#########################################################################
#########################################################################
#########################################################################
#########################################################################

##PLOTS THE CALUE OF THE FINAL SHARE PRICE AGAINST NUMBER OF ITERATIONS##

plt.xlabel('number of iterations')

plt.ylabel('Final share price ($)')

plt.savefig('iteration.png', bbox_inches='tight')

#plt.show()  

###################################################################

##PLOTS THE CUMILATIVE DISTRIBUTION FUNCTION#######################

run_num=np.linspace(1,iteration,iteration) #CREATES AXIS

PofS_plot=(float(iteration)-run_num)/float(iteration)

plt.plot(finalsort,PofS_plot)

plt.xlabel('Share price ($)')

plt.ylabel('$P(S)$   (probability)')

plt.title('CDF')
plt.ylim(0, 1)
plt.xlim(0, 700)

plt.savefig('plot.png')

plt.savefig('plot.png', bbox_inches='tight')

plt.show()

###################################################################

###################################################################

###################################################################

######CREATES TIME AXIS (DAYS)

t=[0]

for l in range(days):

    t.append(l)

    #creating the time axis by appending each new day
###################################################################
###################################################################
###plots the very first iteration of the stock  market graph
#plt.plot(t, sprice,label='Python model of share price')

plt.legend(loc='upper right')

plt.xlabel('number of days')

plt.ylabel('share price ($)')

plt.savefig('shareprice.png', bbox_inches='tight')

#plt.show()

#print ('the final values of share price over', iteration, 'iterations are:', finalshareprice)


##################################################################

##################################################################

##############PLOTTING BP SHARE PRICE AS A FUNCTION OF DAYS#######


bpar=np.genfromtxt("2000.txt")                   #reads BP data from external file

t1=np.linspace(1,2000,2000)                 #makes the x axis splits it into 6885 segments ftom 1 ro 6885

plt.xlabel('number of days')

plt.plot(t1, bpar)

plt.ylabel('Share price ($)')

plt.title('BP share price')      #CREATES LABEL

plt.savefig('bp.png', bbox_inches='tight')  #SAVES PICTURE OF GRAPH

plt.show()

#######################################################################################################################

######################Kelly criterion##############################

geomean = scipy.stats.mstats.gmean(finalshareprice)

print (geomean)

print ('geo mean is above')



fKelly = min(mu/(sigma*sigma),1)

print ('fkelly is', fKelly)

######finding geo mean#########

####################################################################

####################################################################

finalkellyprice = []    #creates an empty array where i will store the finalkellyprice values   

#f = print (input("enter fraction of money you'd like to re-invest"))

#f = float(f)

f = fKelly

for i in range(iteration):
#####################################################################################################
###################### how much money i'm reinvesting into the market
# initial total amount of money

    mprice=[252.0]              #FRACTION OF MONEY
   

    for x in range(days):
    

       deltas = mu*mprice[x]+sigma*mprice[x]*random.gauss(0.0,1.0)

     
       mprice.append(mprice[x]+fKelly*deltas)


    finalkellyprice.append(mprice[-1])

######################################################################
##PLOTTING BOTH CDF - ONE FOR KELLY AND ANOTHER FOR FINALSHARE PRICE##
kelly_geo = scipy.stats.mstats.gmean(finalkellyprice)

finalkellysort = np.sort(finalkellyprice)   

lab = f

##########PRINTS BOTH CDF CURVES#######

print ('the 2 cdfs are:')

KellyPofS_plot=(float(iteration)-run_num)/float(iteration) #PLOTS THE CUMULATIVE DISTRIBUTION GRAPH
plt.plot(finalkellysort,KellyPofS_plot,color='black',label=lab)
plt.plot(finalsort,PofS_plot,color='red',label='Full amount') #PLOTS THE CUMULATIVE DISTRIBUTION GRAPH FOR KELLY'S CRITERION
plt.xlabel('Share price ($)')
plt.ylabel('$P(S)$   (probability)')
plt.ylim(0, 1)
plt.xlim(0, 700)
plt.title('Comparision of CDF')
plt.legend()  
plt.savefig('compare.png', bbox_inches='tight')
plt.show()

#####finding prob of money being a certain amount####

money = input('how much money do you want to make after 2000 days     ')
money = float(money)
#Prob_frac = print (np.interp(money,finalkellysort,KellyPofS_plot))   #relates x value to y (share price to prob)
probfrac = np.interp(money,finalkellysort,KellyPofS_plot)
#Prob_all = print (np.interp(money, finalsort, PofS_plot ))        #relates x value to y (share price to prob)
proball = np.interp(money, finalsort, PofS_plot )

print ('                                   ')
print ('at the end of 2000 days, the prob of the model yielding a final share value being at least '+str(money)+' dollars is', proball*100, '%' )
print ('                                   ')
print ('at  the end of 2000 days, the prob of the model yielded a final share value of being at least ' +str(money)+' dollar if you invest only a fraction of the inital share price is', probfrac*100, '%')

#################################################################################
##################################################################################
#################################################################################
#################################################################################

endcash = geomean - 252
endcashar = mean - 252

if endcash > 0:
    print ('yay you made a profit of ', endcash)
elif endcash < 0:
    print ('aw you lost ', endcash)

############################################################################################################
######################GRAPH OF ERRORS#######################################################################
############################################################################################################
############################################################################################################
################## aiming to plot the difference in price on each day etween the model and the real stock againts a function of time####

pricelist = np.array(sprice).tolist()           #converts sprice array into list
bplist = np.array(bpar).tolist()                #converts bp array into list
#print (bplist)

errorlist = []                                  #creates empty array called errorlist

for i in range(len(pricelist)):
    errorlist += [(((pricelist[i]-bplist[i])/bplist[i])*100.0)]          #calculates percentage error (difference between model list and bp list)
    


plt.plot(t, errorlist, color = 'green')
plt.xlabel('number of days')
plt.title('%Error between the simulation and the BP stock prices for  2000 days')
plt.ylabel('Percentage difference')      
plt.savefig("pererror.png", bbox_inches="tight")              #plots percentage error against time 
plt.show()