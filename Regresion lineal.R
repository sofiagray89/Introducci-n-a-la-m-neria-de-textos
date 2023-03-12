#INSTALAR PAQUETES
#install.packages("qdab")
#install.packages("TextForecast")
#install.packages("LiblineaR")

# CARGAR LIBRERIAS #
library(readxl)
library(NLP)
library(tm)
library(qdap)
library(wordcloud)
library(usethis)
library(tidyverse)
library(devtools)
library(tidymodels)
library(textrecipes)
library(LiblineaR)

# LLAMAR BASE #
data1 = read.csv("movies.csv")
#ruta <- file.choose()
text <- data1$overview
data1<- as.data.frame(data1)
colnames(data1)

# Un histograma sobre el número de peliculas por ranking (calificación)
# tendremos que tener en cuenta el sesgo al elegir un modelo y comprender nuestros resultados.
data1 %>%
  mutate(puntaje = as.numeric(vote_average)) %>%
  count(puntaje) %>%
  ggplot(aes(puntaje, n)) +
  geom_col() +
  labs(x = "Calificación", y = "Número de peliculas por calificación")

# Dividir nuestra base de datos en datos de prueba y datos de entrenamiento
set.seed(1234)
movies <- data1 %>%
  mutate(puntaje = as.numeric(vote_average),
         text = str_remove_all(overview, "'")) %>%
  initial_split()

movies_train <- training(movies)
movies_test <- testing(movies)

# Procesar los datos
final_rec <- recipe(puntaje ~ text, data = movies_train) %>%
  step_tokenize(text, token = "ngrams", options = list(n = 2, n_min = 1)) %>%
  step_tokenfilter(text, max_tokens = tune()) %>%
  step_tfidf(text) %>%
  step_normalize(all_predictors())
final_rec
#Vamos a construir una pequeña función auxiliar de envoltura de recetas para que podamos
#pasar un valor a .stopword_namestep_stopwords()

#stopword_rec <- function(stopword_name) {
#  recipe(puntaje ~ text, data = movies_train) %>%
#    step_tokenize(text) %>%
###########   step_stopwords(text, stopword_source = stopword_name) %>%
#    step_tokenfilter(text, max_tokens = 1e3) %>%
#    step_tfidf(text) %>%
#    step_normalize(all_predictors())
#}
#stopword_rec(snowball)


# Especificar el modelo
svm_spec <- svm_linear() %>%
  set_mode("regression") %>%
  set_engine("LiblineaR")
svm_spec

# combinar la receta de preprocesamiento y la especificación del modelo en un flujo de trabajo ajustable.
tune_wf <- workflow() %>%
  add_recipe(final_rec) %>%
  add_model(svm_spec)
tune_wf

# AJUSTAR EL MODELO #
# Configurar un conjunto de posibles valores de parámetros para probar.
final_grid <- grid_regular(
  max_tokens(range = c(1e3, 6e3)),
  levels = 6   # combinación para 6 modelos
)
final_grid

# remuestreo para la validacion del modelo (validación cruzada de 10 veces)
set.seed(123)
movies_folds <- vfold_cv(movies_train)
movies_folds

# Guardamos las predicciones para poder explorarlas con más detalle, y también 
# se establecen métricas personalizadas en lugar de usar los valores predeterminados.
# Se calcula RMSE, error absoluto medio y error porcentual absoluto medio durante el ajuste.
final_rs <- tune_grid(
  tune_wf,
  movies_folds,
  grid = final_grid,
  metrics = metric_set(rmse, mae, mape),
  control = control_resamples(save_pred = TRUE)
)
final_rs

# EVALUAR EL MODELO #
final_rs %>%
  collect_metrics() %>%
  ggplot(aes(max_tokens, mean, color = .metric)) +
  geom_line(size = 1.5, alpha = 0.5) +
  geom_point(size = 2, alpha = 0.9) +
  facet_wrap(~.metric, scales = "free_y", ncol = 1) +
  theme(legend.position = "none") +
  labs(
    x = "Número de tokens",
    title = "Rendimiento SVM lineal en el número de tokens",
    subtitle = "El rendimiento mejora a medida que incluimos más tokens"
  )

# Elegir un modelo más simple con menos tokens que brinde un rendimiento cercano al mejor;
# por porcentaje de pérdida en comparación con el mejor modelo, con un límite de pérdida del 3%.
chosen_mae <- final_rs %>%
  select_by_pct_loss(metric = "mae", max_tokens, limit = 3)
chosen_mae

# Finalizar nuestro flujo de trabajo ajustable anterior, actualizándolo con este valor. 
final_wf <- finalize_workflow(tune_wf, chosen_mae)
final_wf

# Usar la función para ajustar nuestro modelo por última vez en nuestros datos de entrenamiento y evaluarlo en nuestros datos de prueba.
final_fitted <- last_fit(final_wf, movies)
collect_metrics(final_fitted)


#############################################
# Podemos usar este resultado final para entender cuáles son las variables más importantes en las predicciones.
movies_fit <- extract_fit_parsnip(final_fitted$.workflow[[1]])

movies_fit %>%
  tidy() %>%
  filter(term != "Bias") %>%
  mutate(
    sign = case_when(estimate > 0 ~ "Más tarde (después del año medio)",
                     TRUE ~ "Anterior (antes del año medio)"),
    estimate = abs(estimate),
    term = str_remove_all(term, "tfidf_text_")
  ) %>%
  group_by(sign) %>%
  top_n(20, estimate) %>%
  ungroup() %>%
  ggplot(aes(x = estimate,
             y = fct_reorder(term, estimate),
             fill = sign)) +
  geom_col(show.legend = FALSE) +
  scale_x_continuous(expand = c(0, 0)) +
  facet_wrap(~sign, scales = "free") +
  labs(
    y = NULL,
    title = paste("Importancia variable para predecir el año de",
                  "Opiniones de la Corte Suprema"),
    subtitle = paste("Estas características son las más importantes",
                     "Al predecir el año de una opinión")
  )

# Examinar cómo se comparan los puntajes reales y previstos para el conjunto de pruebas. 
final_fitted %>%
  collect_predictions() %>%
  ggplot(aes(puntaje, .pred)) +
  geom_abline(lty = 2, color = "gray80", size = 1.5) +
  geom_point(alpha = 0.3) +
  labs(
    x = "Puntaje real",
    y = "Puntaje previsto",
    title = paste("Puntajes previstos y verdaderos para el conjunto de pruebas de",
                  "Resumen de las peliculas"),
    subtitle = "Para el conjunto de pruebas, las predicciones son más confiables después de 1850"
  )

# Unamos las predicciones en el conjunto de pruebas con los datos originales de la prueba de opinión de la Corte Suprema y filtremos las observaciones con una predicción que es más de 25 años errónea.
scotus_bind <- collect_predictions(final_fitted) %>%
  bind_cols(scotus_test %>% select(-year, -id)) %>%
  filter(abs(year - .pred) > 25)
# ¿cómo se ven las opiniones más recientes que se predijeron incorrectamente?
scotus_bind %>%
  arrange(-year) %>%
  select(year, .pred, case_name, text)

# Hay algunos ejemplos interesantes aquí donde podemos entender por qué el modelo predeciría mal:
# Vale la pena dedicar tiempo a mirar ejemplos para los que su modelo no funciona bien, por razones similares a las que el análisis exploratorio de datos es valioso antes de comenzar a entrenar su modelo.


