#Group members: 
#1. Andreas Elifas, student no: 218024142
#2. Eliaser Werner, student no: 218062958

using DataFrames,Statistics,CSV,Plots

#loading the data into julia
df = CSV.read("C:\\Users\\digits\\Documents\\Projects\\Julia_Projects\\AIG_Assignment\\bank-additional-full.csv");

#_____________________
#step 1: Data cleaning
#_____________________

#deleting unneeded columns with proper names
result1 = select!(df,Not([:age,:marital,:loan,:housing,:education,:contact,:month,:day_of_week,:duration,
:pdays,:previous,:default,:campaign,:euribor3m]))

#deleting columns with names sapareted by dots
result2 = select!(df,Not([:"emp.var.rate",:"cons.price.idx",:"cons.conf.idx",:"nr.employed"]))

#renaming columns
rename!(df,Dict(:y => :placed_deposit))

#converting string to int
convert_string(str) = try parse(Int64,str) catch;
    if str == "yes"
        return 1
    end # if
    if str == "no"
        return 0
    end
    return missing
end;


#converting the poutcome column
convert_poutcome(str) = try parse(Int64,str) catch
    if str == "nonexistent"
        return 2
    end
    if str == "sucess"
        return 1
    end
    if str == "failure"
        return 0
    end
    return missing
end

#converting the job column to int
#admin = 0,blue-collar = 1,entrepreneur = 2,housemaid = 3,management = 4,retired = 5,
#self-employed = 6,services = 6,student = 7, technician = 9,unemployed = 10
convert_str(str) = try parse(Int64,str) catch;
    if str == "admin"
        return 0
    end
    if str == "blue-collar"
        return 1
    end
    if str == "entrepreneur"
        return 2
    end
    if str == "housemaid"
        return 3
    end
    if str == "management"
        return 4
    end
    if str == "retired"
        return 5
    end
    if str == "self-employed"
        return 6
    end
    if str == "services"
        return 7
    end
    if str == "student"
        return 8
    end
    if str == "technician"
        return 9
    end
    if str == "unemployed"
        return 10
    end
    return missing
end

#passing columns that need to be converted
df.placed_deposit = map(convert_string,df.placed_deposit)
df.poutcome = map(convert_poutcome,df.poutcome)
df.job = map(convert_str,df.job)

#selecting all the complete rows(without missing values)
df2 = df[completecases(df),:]
println(first(df2,10))

#___________________________________________________
#step 2: Spliting data into testing and training set
#___________________________________________________

#creating a vector of x data
x = df2[:,1:2]

#creating a vector y data
y = df2[:,3]

#converting x data into a matrix
xmatrix = convert(Matrix,x)

#X training and testing set
xTrain = xmatrix[1: 24303, :]
xTest = xmatrix[1: 6076, :]

#Y training and testing set
yTrain = y[1:24303, :]
yTest = y[1:6076, :]

#__________________________
#Step 3: Building the model
#__________________________

#function for normalizing the training matrix
function normalize_train(x)
    xmean = mean(x, dims = 1)
    xstd = std(x, dims = 1)

    x_normal = (x .- xmean) ./ xstd
    result = (x_normal,xmean,xstd)

    return result;
end

#normalizing the testing matrix
function normalize_test(x,xmean,xstd)
    x_normal = (x .- xmean) ./ xstd

    return x_normal;
end

#calling the normalize_train function
normalized_train,xmean,xstd = normalize_train(xTrain)

#calling the normalize_test function
normalized_test = normalize_test(xTest,xmean,xstd)

#Sigmoid function
function sigmoid(z)
    result = 1 ./ (1 .+ exp.(.-z))
    return result;
end

#hypothesis function
function hypothesis(x,theta)
    h = sigmoid(x*theta)
    return h
end

#creating a regularization cost function
function regularization(x,y,theta,lamda)
    m = length(y)

    #cross entropy for y = 1 and y = 0
    one_cost = ((-y)' * log.(hypothesis(x,theta)))
    zero_cost = ((1 .- y)' *log.(1 .- hypothesis(x,theta)))

    #adding lambda
    apply_lamda = (lamda/(2*m) * sum(theta[2 : end] .^ 2))

    #calculating the regularization cost
    r_cost = (1/m) * (one_cost - zero_cost) + apply_lamda

    #gradient without constant
    gradient = (1/m) * (x') * (hypothesis(x,theta) - y) + ((1/m)*(lamda*theta))
    gradient[1] = (1/m) * (x[:,1])' * (hypothesis(x,theta) - y)

    return (gradient,r_cost)
end

#____________________________
# Step 3 Peformance matrics
#____________________________

#prediction function
function predict(x,theta)
    m = size(x)[1]
    intercept = true

    if intercept
        cons = ones(m,1)
        x = hcat(cons,x)
    else
        x
    end

    return sigmoid(x * theta)
end

#probability prediction
function probability(p)
    return p .>= 0.5
end

#training and validation
train_result = mean(yTrain .== probability(predict(normalized_train,0.5)))
test_result = mean(yTest .== probability(predict(normalized_test,0.5)))

println("Train result: ",round(train_result,sigdigits = 4))
println("Test results: ",round(test_result,sigdigits = 4))

#confusion matrix
function confusionmat(predicted,actual,d)
    a = zeros(d,d)

    for i in 1:length(actual)
        a[actual[i]+1 ,predicted[i]+1] += 1
    end

    return a;
end

#accuracy function
function accuracy(predicted,actual)
    sum(predicted .== actual) / length(actual)
end

#precision function
function precisn(predicted,actual)
    TP = 0
    FP = 0.

    for i in 1:length(actual)
        if actual[i] == 1
            TP += predicted[i]
        else
            FP += predicted[i]
        end
    end

    return TP / (TP+FP)
end

#recall function
function recall(predicted,actual)
    TP = 0
    FN = 0.

    for i in 1:length(actual)
        if actual[i] == 1
            TP += predicted[i]
            FN += 1.
        end
    end
    return TP / (TP+FN)
end


println(" ")


actual = y
predicted = test_result

println("Confusion Matrix: ",confusionmat(predicted,actual,2))
println("Accuracy: ",accuracy(predicted,actual))
println("Precision: ",precisn(predicted,actual))
println("Recall: ",recall(predicted,actual))
