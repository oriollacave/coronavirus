import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy
from sklearn.linear_model import LinearRegression

## PATHS
basepath="/home/oriol/work/coronavirus"
datapath=basepath+"/data"

#OPTIONS
plot=0
debug=0
#minimum number of cases (last day value > minfilter)
mindeathsfilter=50
#PARAMETRIC VALUES
deathRatio=0.01
detectionRatio=0.05
#COVID FILE TO IMPORT
filecovid="time_series_19-covid-Deaths-nonZero.csv"
filecovidconfirmed="time_series_covid19_confirmed_global.csv"
filecoviddeaths="time_series_covid19_deaths_global.csv"
filecovidrecovered="time_series_covid19_recovered_global.csv"

print ("Data path:"+basepath)

### PLAY CODE OUTSIDE FUNCTIONS.
def main():
	print("  READ   DATA  ")
	minfilter=mindeathsfilter/detectionRatio
	dfc=importCovidData(filecovidconfirmed,minfilter)
	minfilter=mindeathsfilter
	dfd=importCovidData(filecoviddeaths,minfilter)
	minfilter=mindeathsfilter
	dfr=importCovidData(filecovidrecovered,minfilter)
	dft=importMeteoData("T.txt")
	dftm=dft.mean()
	dftt=dftm.transpose()
	dfttdf=pd.DataFrame(dftt)
	dfttdf.columns=['Temperature']
	dftt=dfttdf.copy()
	
	print(" ############################################### ")
	print("   PLAY STUFF")
##################################
## COMBINE METEO (dft)  / COVID 19 (dfc) data
##################################
##################################
#filter relavant days where it has started really
	print("SHIFT")
	dfcs=shift(dfc)
	dfds=shift(dfd)
	dfrs=shift(dfr)
### analysis
	print("MOST LIKELY SHIFT FOR EACH COUNTRY")
	dfmostlikelyshifts=countriesAnalysis(dfc,dfd,dfr).set_index(['country'])
	if plot == 1:
		print("PLOT ALL COUNTRIES")
### plot countries together
		plotCountries(dfds,'deaths')
		plotCountries(dfcs,'confirmed')
		plotCountries(dfrs,'recovered')

#### COUNTRY PLOTS
	print("LOOP COUNTRIES--> detectionRatio and maxR0")

#OPTIONS: ALL OR MANUALLY SELECTED COUNTRIES
#A)SELECTED MANUALLY LIST OF COUNTRIES
	countries=['Spain','US','France','Germany','Italy']
#B)ALL COUNTRIES (that pass filter condition)
	countries=dfmostlikelyshifts.index.values
	print(countries)
	detectionRatios=np.empty(0)
	maxRs=np.empty(0)
	for country in countries:
		mlshift=dfmostlikelyshifts.loc[country,'mostlikelyshift'].astype(int)
		mlshift=7
#TODO DATAFRAME DETECTION RATIO
		detectionRatiotmp,maxRtmp=countryPlots(dfc,dfd,dfr,country,mlshift)
		maxRs=np.append(maxRs,maxRtmp)
		detectionRatios=np.append(detectionRatios,detectionRatiotmp)
		
	data = {'country':countries,'detectionRatio':detectionRatios,'maxR0':maxRs}
	print(data)
	dfdetectionRatios=pd.DataFrame(data)

	dfdetectionRatios=dfdetectionRatios.set_index('country')
	df=pd.concat([dfmostlikelyshifts,dfdetectionRatios,dftt],axis=1,join='inner')
	print(df)
	plt.close()
	pal=sns.color_palette("RdBu", n_colors=7)
	cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
	sns.scatterplot(data=df,x='detectionRatio',y='maxR0',hue='Temperature',palette=cmap)
	plt.title("MAX R0 vs detectionRatio")
	plt.show()

	for column in ['mostlikelyshift','detectionRatio','Temperature']:
		ax = sns.scatterplot(data=df,x=column,y='maxR0')
		plt.title("MAX R0 vs "+column)
		plt.show()
	exit()
	


##################################
	#Get max increase day
	print("MAX INCREASE")
#THE value of this day minus 3 (days between simptoms and infection) + 10 (8th day with simptoms seems most critical) is the date to look for:DATES where virus increased more relatively.
	dfpcntchange=dfs.pct_change()*100
	dfpcntchange=dfpcntchange.replace([np.inf, -np.inf],np.nan)
	dfpcntchange=dfpcntchange.replace([0],np.nan)
#	dfspline=dfpcntchange.interpolate(method='polynomial',order=9)
#plot increasings 
	dfmelt=pd.melt(dfpcntchange.reset_index(),id_vars='index')
	#plt.close()
	ax = sns.lineplot(x='index',y='value',hue='site',data=dfmelt)
#	plt.ylim(0,1)
	plt.title("Increase "+filecovid)
	plt.show()
########
	print("DEATHS vs TEMPERATURE Scatter")
	plt.close()
#plot deaths vs temperature

	t=dft.mean()
	d=dfc.max()

	dfx=dfc.apply(lambda x: x[ x < 100 ], axis=0)
	dfxx=dfc.apply(lambda x: x[ x >1 ], axis=0)
	print(dfxx.head())
	dfxxx=dfxx.reset_index()
	
	dfxxs=dfxx.apply(lambda row: getdaysgrow(row),axis=0)
	print(dfxxs)
	exit()

	t.name=None
	d.name=None

	df=pd.concat([t,d],axis=1,join='inner')
	df.columns=['Temperature','MaxDeaths']
	ax = sns.scatterplot(data=df,x='Temperature',y='MaxDeaths')
	plt.title("AVG TEMP VS MAX Deaths")
	plt.show()

	exit()

	plt.title("Deaths vs Temperature")
	plt.show()
### XY mean temp vs max increase
	print("MEAN TEMP vs INCREASE RATE")
	t=dft.max()
	c=dfpcntchange.mean()
	c.name=None
	t.name=None
	plt.close()
	df=pd.concat([t,c],axis=1,join='inner')
	df.columns=['Temperature','MaxIncreaseRate']
	print(df.describe())
	print(df.head())
	ax = sns.scatterplot(data=df,x='Temperature',y='MaxIncreaseRate')
	plt.title("AVG TEMP VS MAX INCREASE")
	plt.show()



###########################################################################
###########################################################################
##############     PLAY STUFF PLOT; STATS; ETC              ###############
###########################################################################
###########################################################################

###########################################################################
##############     PLOTS COMPARING COUNTRIES                ###############
###########################################################################
# TO PLOT SIMPLE GRAPH VS TIME
def plotCountries(dfs,label):
	dfmelt=pd.melt(dfs.reset_index(),id_vars='index')
	plt.close()
	ax = sns.lineplot(x='index',y='value',hue='site',data=dfmelt)
	plt.title("Number of "+label)
	plt.show()
	plt.close()


###########################################################################
##############     ANALYSE EACH COUNTRY DATA AND COMBINE    ###############
###########################################################################

def countriesAnalysis(dfc,dfd,dfr):
#COUNTRIES ANALYSYS
	print("################ LOOP COUNTRIES >>>>>>>>>")
	countries=dfc.columns
	processedcountries=np.empty(0)
	mlshifts=np.empty(0)
	for country in countries:
		if not country in dfc.columns:
			continue
		if not country in dfd.columns:
			continue
		if not country in dfr.columns:
			continue
		dfcc=dfc[country]
## TODO  SET SAME FILTER
		dfdc=dfd[country]	
		dfrc=dfr[country]	
		df=pd.concat([dfrc,dfdc,dfcc],axis=1,join='inner')
		df.columns=['recovered','deaths','confirmed']
##PLOT LINES
		dfmelt=pd.melt(df.reset_index(),id_vars='DATETIME')
		plt.close()
		ax = sns.lineplot(x='DATETIME',y='value',hue='variable',data=dfmelt)
		plt.title(country)
		#plt.show()
		plt.close()
#linear fit
		dfxy=df[['deaths','confirmed']].dropna()
		X = dfxy.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
		Y = dfxy.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
		linear_regressor = LinearRegression()  # create object for the class
		linear_regressor.fit(X, Y)  # perform linear regression
		correlation=linear_regressor.score(X,Y)
		Y_pred = linear_regressor.predict(X)  # make predictions
		plt.scatter(X, Y)
		plt.plot(X, Y_pred, color='red')
		plt.title(correlation)
		plt.xlabel('deaths')
		plt.ylabel('confirmed')
		#plt.show()
		leng=dfxy['deaths'].size
#compute most likely shift from max R2 
		maxshift=0
		mlshift=0
		for i in range(1,9):
			dfxyn=dfxy.copy()
			newlen=leng-i
			dfxyn['deaths'][0:newlen]=dfxy['deaths'][i:]
			dfxyn['deaths'][newlen:]=np.nan
			dfxyn=dfxyn.dropna()
			X = dfxyn.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
			Y = dfxyn.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
			linear_regressor = LinearRegression()  # create object for the class
			linear_regressor.fit(X, Y)  # perform linear regression
			correlation=linear_regressor.score(X,Y)
			if correlation > maxshift:
				mlshift=i
				maxshift=correlation
			#compute most likely array for countries
			Y_pred = linear_regressor.predict(X)  # make predictions
			plt.close()
			plt.scatter(X, Y)
			plt.plot(X, Y_pred, color='red')
			plt.title("Shift:"+str(i)+" --> R2:"+str(correlation))
		#	plt.show()
		#	print("Shift:"+str(i)+" --> R2:"+str(correlation))
		mlshifts=np.append(mlshifts,mlshift)
		processedcountries=np.append(processedcountries,country)
		
###############m  TODO  remove plots. get max R2 shift. print max shift
	print("################ LOOP COUNTRIES END <<<<<<<<<<<<<<")

### SUMMARY OF MOST LIKELY SHIFT BETWEEN DEATHS AND CONFIRMED ##
	print("####### MOST LIKELY SHIFT DEATHS <----> CONFIRMED   #######")
	data = {'country':processedcountries,'mostlikelyshift':mlshifts}
	dfmlshifts=pd.DataFrame(data)
	return(dfmlshifts)

###########################################################################
##############        COUNTRY PLOTS          ##############################
###########################################################################
def countryPlots(dfc,dfd,dfr,country,nshift):
######################
## typical graph lines
 ## TODO  SET SAME FILTER
#	dfdc=dfd[country].copy()
	dfs=shift(dfd)
	if  dfs[country].size < 1 :
		return np.nan,np.nan
	last_idx = dfs[country].last_valid_index()
	length= dfs[country].size
	if last_idx != None:
		shiftdays=length-last_idx
		dfdc=dfd[country].copy().iloc[shiftdays:]
	else:
		dfdc=dfd[country].copy()
	if  dfdc.size < 20 :
		return np.nan,np.nan
	dfcc=dfc[country]
	if dfcc.size < 20:
		return np.nan,np.nan
	dfrc=dfr[country]
	if dfrc.size < 20:
		return np.nan,np.nan
	df=pd.concat([dfdc,dfrc,dfcc],axis=1,join='inner')
	df.columns=['deaths','recovered','infectedDetected']
	#segons Oriol Mitjà 5% 
	#calculem segons l'última dada
#segons ratio morts 1%
	#shift it 8 days(most likely lag between confirming and death)
	
	dfxyn=df.copy()
	leng=dfxyn['deaths'].size
	newlen=leng-nshift
	dfxyn['deaths'][0:newlen]=df['deaths'][nshift:]
	dfxyn['deaths'][newlen:]=np.nan
	df['infectedEstimatedFromDeaths']=dfxyn['deaths']/deathRatio
	global detectionRatio
	detectionRatio=df['infectedDetected'].iloc[-8]/df['infectedEstimatedFromDeaths'].iloc[-8]
	df['infectedEstimatedFromConfirmed']=df['infectedDetected']/detectionRatio

#TODO compute deaths forecast based on infected(confirmed) and shifting 7 days ahead
	#shit this 7 days ahead (forecast)
	deathsForecast=df['infectedEstimatedFromConfirmed'].copy().reset_index().drop(['DATETIME'],1)*deathRatio
	deathsForecast.columns=['deathsForecast']
	data={'deathsForecast':[]}
	kk=pd.DataFrame(data)
	kkdf=df.iloc[1].copy()
	kkdf[:]=np.nan
	for n in range(nshift):
		kk=kk.append({'deathsForecast':np.nan},ignore_index=True)
		df=df.append(kkdf,ignore_index=True)

	deathsForecast=kk.append(deathsForecast,ignore_index=True)
	df=pd.concat([df,deathsForecast],axis=1,join='inner')
	if plot == 1:
		plotLinesCountry(df,'Aggregates',country)
#compute daily values, not aggregates
	dfincremental=df.diff()
	if plot == 1:
		plotLinesCountry(dfincremental,'dailyValues',country)

#Compute R parameter
	dfR=computeRdf(dfincremental)
	print(dfR)
## TODO make 5 max, average 4 not max of it, return
	maxRs=dfR.nlargest(10,'deaths')['deaths'][2:-2].mean()
	print(country)
	print(dfR)
	print(maxRs)
	if plot == 1:
		plotLinesCountry(dfR,'Rparameter',country)
	return detectionRatio,maxRs



def plotLinesCountry(df,tag,country):
	dfgraph=df[['infectedDetected','infectedEstimatedFromDeaths','infectedEstimatedFromConfirmed']]
##PLOT LINES
	dfmelt=pd.melt(dfgraph.reset_index(),id_vars='index')
	plt.close()
	fig = plt.figure(frameon = False)
	fig.set_size_inches(8, 8)
	ax = sns.lineplot(x='index',y='value',hue='variable',data=dfmelt)
	pct=int(detectionRatio*100)
	plt.title(tag+" "+country+" | Death Ratio:"+str(deathRatio)+" | Detection ratio: "+str(pct)+"%")
	plt.savefig("img/"+country+"-"+tag+"-infected+estimations.png")
	plt.show()
	plt.close()

	dfgraph=df[['deaths','deathsForecast']]
##PLOT LINES
	dfmelt=pd.melt(dfgraph.reset_index(),id_vars='index')
	plt.close()
	fig = plt.figure(frameon = False)
	fig.set_size_inches(8, 8)
	ax = sns.lineplot(x='index',y='value',hue='variable',data=dfmelt)
	pct=int(detectionRatio*100)

	plt.title(tag+" "+country+" | Death Ratio:"+str(deathRatio)+" | Detection ratio: "+str(pct)+"%")
	plt.savefig("img/"+country+"-"+tag+"-deaths+forecast.png")
	plt.show()
	plt.close()




###########################################################################
##############       AUXILIARY FUNCTIONS     ##############################
###########################################################################

def shift(dfc):
#filter relavant days where it has started really
	dfx=dfc.apply(lambda x: x[ x > 10 ], axis=0)
	#STUF
	dfx=dfx.reset_index()
	dfx=dfx.drop(['DATETIME'],1)
#manipulate to set all starting at same days. Move NaN to the end.
	dfs=dfx.apply(lambda row: shiftrow(row),axis=0)
	return dfs

def shiftrow(row):
	rowa=row.copy()
	if np.isnan(rowa.iloc[-1]):
		return rowa
	else:
		rowa.loc[:]=np.nan
		first_idx = row.first_valid_index()
		last_idx = row.last_valid_index()
		valids=int(last_idx)-int(first_idx)
		rowa.loc[0:valids]=row.loc[first_idx:].values
		return rowa

def computeRdf(df):
	dfs=df.apply(lambda row: computeR(row),axis=0)
	return dfs

def computeR(row):
	length= row.size
	rowR=row.copy()
	for i in range(length):
		if (i > 3):
			n=i-4
			rowR[i]=row[i]/row[n]
		else:
			rowR[i]=np.nan

	return rowR

def getdaysgrow(row):
	first_idx = row.first_valid_index()
	last_idx = row.last_valid_index()
	valids=int(last_idx)-int(first_idx)
	return valids

###########################################################################
###########################################################################
##############        READ /  IMPORTS                       ###############
###########################################################################
###########################################################################

def importCovidData(filecovid,minfilter):
#############################################
# COVID19 Data read
# Province/State,Country/Region,Lat,Long,1/22/20,1/23/20,1/24/20,1/25/20,1/26/20,1/27/20,1/28/20,1/29/20,1/30/20,1/31/20,2/1/20,2/2/20,2/3/20,2/4/20,2/5/20,2/6/20,2/7/20,2/8/20,2/9/20,2/10/20,2/11/20,2/12/20,2/13/20,2/14/20,2/15/20,2/16/20,2/17/20,2/18/20,2/19/20,2/20/20,2/21/20,2/22/20,2/23/20,2/24/20,2/25/20,2/26/20,2/27/20,2/28/20,2/29/20,3/1/20,3/2/20,3/3/20,3/4/20,3/5/20,3/6/20,3/7/20,3/8/20,3/9/20,3/10/20,3/11/20,3/12/20,3/13/20,3/14/20,3/15/20,3/16/20,3/17/20,3/18/20,3/19/20
#############################################
#CHANGE THIS TO GLOBAL
	print("####################################################################")
	print("##############				   PROCESSING "+filecovid+" DATA............")
	covid19file=datapath+"/"+filecovid
	dfc = pd.read_csv(covid19file, sep=',')
	df1 = dfc.replace(np.nan, '', regex=True)
	dfc=df1.replace(' ', '', regex=True)
	df1 = dfc.replace(0, np.nan, regex=True)
	dfc=df1
	dfc['site']=dfc['Province/State'].map(str)+dfc['Country/Region'].map(str)
########################## SET FILTER FOR COVID PLACES !!!!!
#filter
	cols=list(dfc.columns)
	dfc=dfc[dfc[cols[-2]] > minfilter]
	sites=dfc['site']
	dfc=dfc.drop(['site','Province/State','Country/Region','Lat','Long'],1)
#prepare
	dfct=dfc.transpose()
	dfct.index.name = None
	dfc=dfct
	cols=list(dfc.columns)
#to pcnt
#dfc[cols] = dfc[cols].div(dfc[cols].sum(axis=1), axis=0).multiply(100)
	dfc.columns=sites
	dfc.index.name = None
### TO DATETIME
	dfc['DATETIME']=dfc.index
	dfc['DATETIME']=pd.to_datetime(dfc['DATETIME'],format='%m/%d/%y')
	dfc.reset_index(drop=True,inplace=True)
	dfc.set_index(['DATETIME'],inplace=True)
	if debug == 1:
		print("########## DEBUG  #########")
		print(dfc.describe())
		print(dfc.head())
	if plot == 1:
		print("########## PLOT  #########")
		dfcmelt=pd.melt(dfc.reset_index(),id_vars='DATETIME')
		#plot
		ax = sns.lineplot(x='DATETIME',y='value',hue='site',data=dfcmelt)
		plt.show()
	return dfc


def importMeteoData(filemeteo):
#########################################
###	 READ  METEO
#########################################
	print("####################################################################")
	print("#################			  PROCESSING METEO "+filemeteo+"............")
	print ("Data path:"+basepath)
#temperature read
# yyyymmddHHMM site1 site2 .... siten
	meteofile=datapath+"/"+filemeteo
	df = pd.read_csv(meteofile, sep=' ')
	df[['DATETIME']]=pd.to_datetime(df[['DATETIME']].stack(),format='%Y%m%d%H%M').unstack()
	df.set_index('DATETIME',inplace=True)
	df = df.astype(float)
	dfdayavg=df.resample('D').mean()
	dft=dfdayavg
	if plot==1:
		print("########## PLOT  #########")
		dfmelt=pd.melt(dfdayavg.reset_index(),id_vars='DATETIME')
		ax = sns.lineplot(x='DATETIME',y='value',hue='variable',data=dfmelt)
		plt.show()
		dftelted=dfmelt
	if debug==1:
		print("########## DEBUG  #########")
		print(dft.describe())
		print(dft.head())
	return dft

############################################################################
##                   EXECUTE >>>>>>>>>>>>
main()
##                   <<<<<<<<<<<< EXECUTE
############################################################################
