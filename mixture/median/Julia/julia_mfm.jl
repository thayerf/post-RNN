# A basic example of using the BayesianMixtures package.

using BayesianMixtures
B = BayesianMixtures
# Open our file
using CSV
df = CSV.read("data.csv", header =false, types = fill(String,250))
labels = CSV.read("labels.csv",header = false,types = fill(String,1))
y = map(x-> parse(Float64,x),convert(Array,labels[:,1])[:,1])
##

function get_tmax(results)
    tmax = fill(-Inf,size(results)[2])
    for i in 1:size(results)[2]
        for j in 1:43
            if results[j,i][2]==0
            else
                if tmax[i]<results[j,i][1]
                    tmax[i]=results[j,i][1]
                end
            end
        end
    end
    return tmax
end
preds = fill(0.0,500)
for i in 1:500
    x = map(x->parse(Float64,x),convert(Array,df[i,:])[1,:])

    # Specify model, data, and MCMC options
    n_total = 2000  # total number of MCMC sweeps to run
    options = B.options("Normal","MFM",x,n_total, log_pk = "k -> -5+(k-1)*log(5)-log(factorial(big(k-1)))", t_max = 40)
    # Run MCMC sampler
    result = B.run_sampler(options)
    ##
    # Results no burn-in
    results = [result.theta[i,j]for i in 1:43,j in 1000:2000]
    preds[i]= quantile(get_tmax(results),0.5)
end

mean(abs(preds-y))
using DataFrames
CSV.write("juliapreds.csv",DataFrame(preds);append=false)
