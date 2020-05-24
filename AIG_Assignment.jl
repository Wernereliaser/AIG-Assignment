import Pkg
Pkg.add("Plots")
Pgk.add("MLBase")

using DataFrames, CSV, Statistics, Plots, MLBase
#loading the dataset into julia
df = CSV.read("C:/Users/IG_Assignment/bank-additional-full.csv")

#STEP 1 : Data cleaning

#Deleting un-needed columns with proper names
result1 = select!(df,Not([:age, :job, :education, :contact, :poutcome, :month, :day_of_week, :duration, :pdays, :previous, :default, 
      :campaign, :euribor3m]))

#deleting columns with names separated by dots
result2 = select!(df, Not([:"emp.var.rate", :"cons.price.idx", :"cons.conf.idx", :"nr.employed"]))
