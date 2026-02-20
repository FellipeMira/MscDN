import ee
import math
from datetime import datetime


def build_era5_land_cube(roi, start_date, end_date, time_step='1h', scale=10000, crs='EPSG:4326'):
    """
    Constrói um cubo espaço-temporal utilizando exclusivamente a coleção ERA5-Land nativa do GEE.
    """
    
    # Carregar a coleção ERA5-Land (apenas dados de superfície, resolução nativa de ~9km)
    era5_land = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY") \
                  .filterDate(start_date, end_date) \
                  .filterBounds(roi)

    def process_hourly(img):
        time = img.get('system:time_start')
        
        # --- Variáveis e Conversões ---
        # Precipitação: vem em metros, converter para mm
        tp = img.select('total_precipitation').multiply(1000).rename('tp')
        tp_log = tp.add(1).log().rename('tp_log') # log1p(tp)
        
        # Temperaturas: vêm em Kelvin, converter para Celsius
        t2m = img.select('temperature_2m').subtract(273.15).rename('t2m')
        d2m = img.select('dewpoint_temperature_2m').subtract(273.15).rename('d2m')
        
        # Vento a 10m: m/s (mantém a unidade)
        u10 = img.select('u_component_of_wind_10m').rename('u10')
        v10 = img.select('v_component_of_wind_10m').rename('v10')
        
        # Pressão na superfície: vem em Pascals, converter para hPa (mbar)
        sp = img.select('surface_pressure').divide(100).rename('sp')
        
        # Combinar todas as bandas
        cube_img = ee.Image([tp, tp_log, t2m, d2m, u10, v10, sp])
        
        # Reprojetar para garantir escala e máscara consistente
        cube_img = cube_img.reproject(crs=crs, scale=scale) \
                           .clip(roi) \
                           .unmask(-9999) \
                           .set('system:time_start', time)
                           
        return cube_img

    # Aplica o processamento para todas as imagens horárias
    processed_hourly = era5_land.map(process_hourly)
    
    # --- Lógica de Agregação Temporal ---
    if time_step in ['1d', 'daily']:
        # Calcula quantos dias existem no período
        n_days = ee.Date(end_date).difference(ee.Date(start_date), 'day')
        days_list = ee.List.sequence(0, n_days.subtract(1))
        
        def agg_daily(d):
            day_start = ee.Date(start_date).advance(d, 'day')
            day_end = day_start.advance(1, 'day')
            day_col = processed_hourly.filterDate(day_start, day_end)
            
            # Precipitação diária: Soma
            tp_sum = day_col.select('tp').sum().rename('tp')
            tp_log = tp_sum.add(1).log().rename('tp_log')
            
            # Variáveis instantâneas: Média diária
            instantaneous_vars = ['t2m', 'd2m', 'u10', 'v10', 'sp']
            others_mean = day_col.select(instantaneous_vars).mean()
            
            return ee.Image([tp_sum, tp_log, others_mean]).set('system:time_start', day_start.millis())
            
        return ee.ImageCollection(days_list.map(agg_daily))
        
    return processed_hourly

def export_cube_to_drive(image_collection, roi, folder_name, scale, crs, export_type='batch'):
    """
    Exporta o cubo espaço-temporal para o Google Drive.
    """
    if export_type == 'single_image':
        # Útil apenas para recortes temporais pequenos (gera uma banda para cada variável em cada tempo)
        single_image = image_collection.toBands().clip(roi)
        
        task = ee.batch.Export.image.toDrive(
            image=single_image,
            description='ERA5Land_Cube_Single',
            folder=folder_name,
            scale=scale,
            crs=crs,
            region=roi,
            maxPixels=1e13
        )
        task.start()
        print("Exportação toBands iniciada. Verifique as 'Tasks' na UI do Earth Engine (se estiver usando).")

    elif export_type == 'batch':
        # Ideal para ML: exporta um GeoTIFF independente para cada timestep
        count = image_collection.size().getInfo()
        img_list = image_collection.toList(count)
        
        print(f"Iniciando exportação em lote de {count} imagens...")
        for i in range(count):
            img = ee.Image(img_list.get(i))
            
            # Extrair a data para nomear o arquivo
            date_ms = img.get('system:time_start').getInfo()
            date_str = datetime.utcfromtimestamp(date_ms / 1000).strftime('%Y%m%d_%H%M')
            filename = f"ERA5Land_Cube_{date_str}"
            
            task = ee.batch.Export.image.toDrive(
                image=img,
                description=filename,
                folder=folder_name,
                scale=scale,
                crs=crs,
                region=roi,
                maxPixels=1e13
            )
            task.start()
        print("Todas as tarefas de exportação foram enviadas para o GEE!")

# ==========================================
# EXEMPLO DE EXECUÇÃO
# ==========================================
if __name__ == "__main__":
    # 1. Definir ROI (Exemplo: Bounding box englobando a Grande SP)
    roi = ee.Geometry.Rectangle([-47.0, -24.0, -45.5, -23.0])
    
    # 2. Parâmetros (Facilmente editáveis)
    START_DATE = '2023-01-01'
    END_DATE = '2023-01-05'  
    TIME_STEP = '1h'         # Escolha '1h' (horário) ou '1d' (diário)
    SCALE = 10000            # Resolução espacial forçada para exportação (10km)
    CRS = 'EPSG:4326'        # Sistema de coordenadas
    
    # 3. Gerar o Cubo
    cube = build_era5_land_cube(roi, START_DATE, END_DATE, TIME_STEP, SCALE, CRS)
    
    # 4. Validar o resultado imprimindo as bandas do primeiro timestep
    first_image = cube.first()
    print("Bandas extraídas (prontas para ML):", first_image.bandNames().getInfo())
    
    # 5. Exportar (Descomente para testar)
    # export_cube_to_drive(cube, roi, folder_name='ML_Nowcasting_Cube', scale=SCALE, crs=CRS, export_type='batch')