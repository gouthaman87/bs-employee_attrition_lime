# ** Blog Link: https://www.youtube.com/watch?v=NVI4gVfrSZk
# ** Blog Link: https://www.youtube.com/watch?v=zoqFD9-vB-s

# load libraries ----------------------------------------------------------

library(tidyverse)
library(magrittr)
library(skimr)
library(esquisse)
library(GGally)
library(tidyquant)
library(tidymodels)
library(h2o)
library(lime)
ggplot2::theme_set(tidyquant::theme_tq())

# read data ---------------------------------------------------------------

emp_dt <- readr::read_csv("inst/extdata/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# data eda ----------------------------------------------------------------

skimr::skim(emp_dt)

esquisse::esquisser()

emp_dt <- emp_dt %>% 
  # remove variables not involve in ML
  dplyr::select(-c(EmployeeCount, EmployeeNumber, Over18, StandardHours)) %>% 
  
  dplyr::mutate_at(
    .vars = c(
      "Education", "EnvironmentSatisfaction", "JobInvolvement",
      "JobLevel", "JobSatisfaction", "PerformanceRating",
      "RelationshipSatisfaction", "StockOptionLevel", "WorkLifeBalance"
    ),
    ~as.factor(.)
  )
          
emp_dt %>% 
  dplyr::select_if(is.numeric) %>% 
  GGally::ggpairs()

# ML models ---------------------------------------------------------------

set.seed(1987)
splits <- rsample::initial_split(emp_dt, strata = Attrition, prop = 0.8)

recipe_spec <- recipes::recipe(Attrition ~ ., data = rsample::training(splits)) %>% 
  recipes::step_zv(recipes::all_predictors()) %>% 
  recipes::step_corr(recipes::all_numeric())

train <- recipe_spec %>% recipes::prep() %>% recipes::juice()

h2o::h2o.init()

y <- "Attrition"
x <- setdiff(colnames(train), y)

automl_model <- h2o::h2o.automl(
  x, 
  y, 
  training_frame = h2o::as.h2o(train), 
  max_runtime_secs = 60*5, 
  stopping_metric = "AUC"
)

automl_model@leaderboard %>% 
  dplyr::as_tibble() %>% 
  dplyr::slice_max(order_by = auc, n = 10) %>% 
  dplyr::select(1:3) %>% 
  tidyr::pivot_longer(cols = 2:3) %>% 
  dplyr::mutate(value = round(value, 2),
                model = stringr::str_extract(model_id, "[^_]+")) %>% 
  
  ggplot2::ggplot(ggplot2::aes(value, model_id, color = model)) +
  ggplot2::geom_point(size = 3) +
  ggplot2::geom_label(ggplot2::aes(label = value)) +
  ggplot2::facet_wrap(~name) +
  tidyquant::scale_color_tq()
    
# Lime --------------------------------------------------------------------

test <- recipe_spec %>% 
  recipes::prep() %>% 
  recipes::bake(rsample::testing(splits))

pred_tbl <- automl_model %>% 
  h2o::h2o.predict(newdata = h2o::as.h2o(test)) %>% 
  dplyr::as_tibble() %>% 
  dplyr::bind_cols(
    test %>% 
      dplyr::select(Attrition)
  )

pred_tbl %$% table(predict, Attrition)

test %>% 
  dplyr::slice(8, 10, 13) %>% 
  dplyr::glimpse()

explainer <- lime::lime(
  train, 
  automl_model@leader, 
  bin_continuous = FALSE, 
  n_bins = 5, 
  n_permutations = 1000
)

class(explainer)

explanation <- test %>% 
  dplyr::slice(1:20) %>% 
  dplyr::select(-Attrition) %>% 
  lime::explain(
    explainer, 
    n_labels = 1, 
    n_features = 10, 
    n_permutations = 5000, 
    kernel_width = 1
  )

lime::plot_features(explanation)

lime::plot_explanations(explanation)

h2o::h2o.shutdown(prompt = F)
