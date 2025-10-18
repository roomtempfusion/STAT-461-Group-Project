devtools::install_github("outbreak-info/R-outbreak-info")
install.packages("miniUI")
install.packages("promises")

library(outbreakinfo)
library(dplyr)
library(knitr)
library(lubridate)
library(ggplot2)
library(purrr)
library(here)
authenticateUser()

alpha_lineages = lookupSublineages("Alpha", returnQueryString = TRUE)
epsilon_lineages = lookupSublineages("Epsilon", returnQueryString = TRUE)
iota_lineages = lookupSublineages("Iota", returnQueryString = TRUE)
delta_lineages = lookupSublineages("Delta", returnQueryString = TRUE)
omicron_lineages = lookupSublineages("Omicron", returnQueryString = TRUE)
beta_lineages = lookupSublineages("Beta", returnQueryString = TRUE)
gamma_lineages = lookupSublineages("Gamma", returnQueryString = TRUE)

who_labels = c("Alpha", "Epsilon", "Iota", "Delta")
names(who_labels) = c(alpha_lineages, epsilon_lineages, iota_lineages, delta_lineages)

il_counties = read.csv('../raw data/il_counties.csv')

il_county_names = paste0("USA_US-IL_", il_counties$county)

splt = unlist(strsplit(omicron_lineages, 'OR')) # get a vector of all variants under omicron umbrella

omicron_1 = paste(splt[1:500], collapse = "OR")
omicron_2 = sub("^\\s", "", paste(splt[501:1000], collapse = "OR"))
omicron_3 = sub("^\\s", "", paste(splt[1001:1500], collapse = "OR"))
omicron_4 = sub("^\\s", "", paste(splt[1501:2000], collapse = "OR"))
omicron_5 = sub("^\\s", "", paste(splt[2001:2109], collapse = "OR"))
# Alternative version of API call that doesn't process location names into ISO codes, instead directly takes ISO codes
getGenomicData_alt <- function(query_url, location=NULL, cumulative=NULL, pangolin_lineage=NULL, mutations=NULL, ndays=NULL, frequency=NULL, subadmin=NULL, other_threshold=NULL, nday_threshold=NULL, other_exclude=NULL, logInfo=TRUE){
  
  genomic_url <- "https://api.outbreak.info/genomics/"
  
  q <- c()
  
  q <- c(q, paste0(query_url), "?")
  
  if(!is.null(location)){
    q <- c(q, paste0("location_id=", location, "&"))
  }
  if(!is.null(cumulative)){
    if (!is.logical(cumulative)){
      stop("cumulative must be in Boolean format")
    }else{
      q <- c(q, paste0("cumulative=", tolower(cumulative)), "&")
    }
  }
  if(!is.null(subadmin)){
    if (!is.logical(subadmin)){
      stop("subadmin must be in Boolean format")
    }else{
      q <- c(q, paste0("subadmin=", tolower(subadmin)), "&")
    }
  }
  if(!is.null(pangolin_lineage)){
    q <- c(q, paste0("pangolin_lineage=", pangolin_lineage, "&"))
  }
  if(!is.null(mutations)){
    check_cond <- grepl("[A-Za-z0-9]+:[A-Za-z][0-9]+[A-Za-z]", mutations)
    if(!all(check_cond))
      warning(paste0("Mutations should be specified in the format gene:mutation, like \"S:E484K\". The following mutations are not in the specified format: ",  paste(mutations[!check_cond], collapse=", ")))
    mutations <- paste(mutations, collapse=" AND ")
    q <- c(q, paste0("mutations=", mutations, "&"))
  }
  if(!is.null(ndays)){
    q <- c(q, paste0("ndays=", ndays, "&"))
  }
  if(!is.null(frequency)){
    q <- c(q, paste0("frequency=", frequency, "&"))
  }
  if(!is.null(other_threshold)){
    q <- c(q, paste0("other_threshold=", other_threshold, "&"))
  }
  if(!is.null(nday_threshold)){
    q <- c(q, paste0("nday_threshold=", nday_threshold, "&"))
  }
  if(!is.null(other_exclude)){
    other_exclude <- paste(other_exclude, collapse=",")
    q <- c(q, paste0("other_exclude=", other_exclude, "&"))
  }
  
  q <- paste(q, sep="", collapse = "")
  q <- sub("&$", "", q)
  
  dataurl <- paste0(genomic_url, q)
  results <- getGenomicsResponse(dataurl, logInfo);
  
  if (length(results) > 1){
    hits <- rbind_pages(results)
  }else{
    hits <- data.frame(results)
  }
  if ("date" %in% colnames(hits)){
    hits$date=as.Date(hits$date, "%Y-%m-%d")
    hits <- hits[order(as.Date(hits$date, format = "%Y-%m-%d")),]
  }
  return(hits)
}

z = getGenomicData_alt(query_url="prevalence-by-location", location="USA_US-IL_17031", pangolin_lineage = "B.1.1.7")

# Alternative version of prevalence call that doesn't process location names into ISO codes, instead directly takes ISO codes
getPrevalence_alt <- function(pangolin_lineage=NULL, location=NULL, mutations=NULL, cumulative=FALSE, logInfo=TRUE){
  if(is.null(pangolin_lineage) && is.null(mutations)) {
    stop("Either `pangolin_lineage` or `mutations` needs to be specified")
  }
  
  if(length(pangolin_lineage) > 1) {
    df <- map_df(pangolin_lineage, function(lineage) getGenomicData_alt(query_url="prevalence-by-location", pangolin_lineage = lineage, location = location, mutations = mutations, cumulative = cumulative, logInfo = logInfo))
  } else {
    df <- getGenomicData_alt(query_url="prevalence-by-location", pangolin_lineage = pangolin_lineage, location = location, mutations = mutations, cumulative = cumulative, logInfo = logInfo)
  }
  
  
  if(!is.null(df) && nrow(df) != 0 && cumulative == FALSE){
    df <- df %>%
      rename(lineage = query_key) %>%
      mutate(location = ifelse(is.null(location), "Worldwide", location))
  }
  
  if(!is.null(df) && nrow(df) != 0 && cumulative == TRUE){
    df <- df %>%
      rename(lineage = key) %>%
      mutate(location = ifelse(is.null(location), "Worldwide", location))
  }
  
  return(df)
}

getPrevalence_alt(omicron_2, location=county)
var = 1
results_2 = data.frame()
for (county in il_county_names){
  for (omicron_var in c(omicron_1, omicron_2, omicron_3, omicron_4, omicron_5)){
    res = getPrevalence_alt(
      pangolin_lineage = omicron_var, location = county)
    results_2 = rbind(results_2, res)
    print(c(county, var))
    var = var + 1
    if (var > 5){
      var == 1
    }
    }
  }

results = data.frame()
for (county in il_county_names){
  for (variant in c(beta_lineages, gamma_lineages)){
    res = getPrevalence_alt(
      pangolin_lineage = variant, location = county)
    results = rbind(results, res)
  }
}


for (county in il_county_names){
  print(county)
}
# results$lineage[results$lineage == alpha_lineages] = 'Alpha'
# results$lineage[results$lineage == delta_lineages] = 'Delta'
# results$lineage[results$lineage == epsilon_lineages] = 'Epsilon'
# results$lineage[results$lineage == iota_lineages] = 'Iota'

results$lineage[results$lineage == beta_lineages] = 'Beta'
results$lineage[results$lineage == gamma_lineages] = 'Gamma'

results_2$lineage[grepl("B.1.1.529", results_2$lineage)] = 'omicron_1'
results_2$lineage[grepl("BJ.1", results_2$lineage)] = 'omicron_2'
results_2$lineage[grepl("FK.1.3.2", results_2$lineage)] = 'omicron_3'
results_2$lineage[grepl("LF.10", results_2$lineage)] = 'omicron_4'
results_2$lineage[grepl("PE.1.3", results_2$lineage)] = 'omicron_5'

unique(results$lineage)

write.csv(results, 'variant_prevalence_2_il.csv', row.names =FALSE)


