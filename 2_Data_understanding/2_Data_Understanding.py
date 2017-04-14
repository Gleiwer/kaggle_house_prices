
# coding: utf-8

# # 2. Data Understanding
# ## 2.1 Collect Initial Data
# ### Outputs:
# - Initial Data Collection Report
# 
# 

# ## 2.2 Describe Data
# ### Outputs:
# - Data Description Report 
# 
# [Data description.txt](./data_description.txt)
# 
# 

# ## 2.3 Explore Data
# 
# [2.3_Explore_Data.ipynb](./2.3_Explore_Data.ipynb)
# 
# ### Conclusions
# 
# #### A descartar
# - La variable utilities, se descartará dado el desbalanceo entre los valores, de 
# los 1460 registros, sólo 1 cuenta con un valor diferente al valor que indica que se
# tienen todos los servicios públicos.
# 
# ### A limpiar
# - De la variable LotArea, se consideran extremos los valores por encima de 55000
# - LotFrontage
#     - La variable LotFrontage, presenta un extremo por encima de 300
#     - 259 Valores nulos, que se deben llenar con valores promedio
# - los 1369 valores nulos de la variable Alley, se deben reemplazar con un valor NA ya que corresponde a información real.
# - MasVnrArea 
#     - La variable MasVnrArea, presenta extremos por encima de 1200
#     - Los valores nulos se deben reemplazar por valores promedio 
# - En BsmtQual, BsmtCond, BsmtExposure BsmtFinType1 BsmtFinType2, se deben reemplazar los valores nulos por NA ya que corresponde a información real
# - La variable BsmtFinSF1, presenta extremos por encima de 5000
# - La variable BsmtFinSF2, presenta extremos por encima de 1400
# - La variable TotalBsmtSF, presenta extremos por encima de 3500
# - La variable Electrical presenta valores nulos.
# - La variable 1stFlrSF, presenta extremos en 0 y por encima de 4000
# - La variable GrLivArea, presenta extremos en 0
# - La variable MiscFeature y MiscVal, se reemplazan los valores nulos por NA y 0 respectivamente
# - En las variables FireplaceQu  y GarageType, e deben reemplazar los nulos por NA, es información real
# - Las variables GarageYrBlt, GarageFinish, GarageQual, GarageCond, se deben reemplazar los nulos por NA
# - La variable WoodDeckSF tiene extremos por encima de 750
# - La variable OpenPorchSF tiene extremos por encima de 400
# - La variable EnclosedPorch, presenta extremos por encima de 400
# - En la variable PoolQC y Fence se debe reemplazar los nulos por NA
# 
# 
# - Se puede construir un nuevo campo fecha_venta a partir de los MoSold y YrSold
# 
# 
# 

# ## 2.4 Verify Data Quality
# ### Outputs:
# 
# - Data Quality Report
