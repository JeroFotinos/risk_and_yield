tima versión: agromodel_model_plantgrowth_v27

Datos de entrada:
- mat_maiz_2021_lat_lowres.mat y mat_maiz_2021_lon_lowres.mat
	Matrices que representan las coordenadas de cada pixel en el área de estudio
	La entrada ij de cada matriz representa el valor de la latitud y longitud del pixel ij
- mat_maiz_2021_lowres.mat
	Matrices que indican con una mascara que pixel del área de estudio se estimó que tenía el tipo de cultivo
	El tamañano debe ser igual al de las matrices mat_maiz_2021_lat_lowres.mat y mat_maiz_2021_lon_lowres.mat
- mat_aguadisp_saocom_maiz_2021-2022_2.mat
	Matriz con datos de nivel de agua disponible como dato inicial
	El tamañano debe ser igual al de las matrices mat_maiz_2021_lat_lowres.mat y mat_maiz_2021_lon_lowres.mat
- fechasiembra_2021-2022.xlsx
	Archivo con información de fechas de siembra
- cargar_coordenadas_maiz2021
	Módulo que lee y transforma la información en el archivo con información de fechas de siembra
- mat_dds_maiz_est_lowres
	Matriz con estimación de días desde siembra que pasaron en cada píxel hasta día de inicio del modelo
- fechas_agua_m y agua_obs_m 
	Datos de fecha y coordenadas geográficas de niveles de agua medida en campo
- ArroyitoDatos2020
	Datos climáticos generales (sin geolocalización)
- clima octubre y noviembre 2021
	Temperaturas máximas y mínimas y radiación (sin geolocalización)
- datosObservados20220117_20220311
Datos climáticos generales observados en estación meteorológica(sin geolocalización)
- EstacionObservado_20220312_20220507
Datos climáticos generales observados en estación meteorológica (sin geolocalización)
- EstacionObservado_20220312_20220807
Datos climáticos generales observados en estación meteorológica (sin geolocalización)
- PAR 301121 al 170122
	Datos climáticos generales observados (sin geolocalización)
	
Datos de salida:
	rend: Tensor con el rendimiento esperado por cada pixel (dimensión 1 y 2) del área en cada tiempo (dimensión 3) 
	rend_final: Matriz con el rendimiento a tiempo final
	rend_esc: Tensor con el rendimiento esperado con la función de escalado aplicada
	RR: Tabla con los valores de cada variable para cada coordenada de campo medido a lo largo del tiempo de simulación
	
Archivos generados:
	datos_maiz: Archivo con la información de la tabla RR


	