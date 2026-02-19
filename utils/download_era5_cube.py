import ee
import geemap
import os
import json
import geopandas as gpd
from datetime import datetime, timedelta

# --- Configurações ---
ROI_FILE = r'data/vector/ValeDoParaiba.geojson'
START_DATE = '2022-12-01'
END_DATE = '2023-03-01'
COLLECTION_ID = 'ECMWF/ERA5_LAND/HOURLY'

# Mapeamento de variáveis GEE para nomes curtos físicos
VARIABLES = {
    'total_precipitation_hourly': 'tp',
    'temperature_2m': 't2m',
    'u_component_of_wind_10m': 'u10',
    'v_component_of_wind_10m': 'v10',
    'surface_pressure': 'sp'
}

def rename_and_prepare(image):
    """
    Renomeia as bandas e ajusta o ID da imagem para facilitar o toBands().
    Formato desejado do prefixo: M{month}_D{day}_{year}_H{hour}
    """
    date = ee.Date(image.get('system:time_start'))
    
    # Formatação de strings no GEE
    year = date.get('year').format('%d')
    month = date.get('month').format('%02d')
    day = date.get('day').format('%02d')
    hour = date.get('hour').format('%02d')
    
    # Criar o prefixo: M12_D10_2024_H01
    prefix = ee.String('M').cat(month).cat('_D').cat(day).cat('_').cat(year).cat('_H').cat(hour)
    
    # Selecionar variáveis e renomear com o sufixo curto
    img_selected = image.select(list(VARIABLES.keys()), list(VARIABLES.values()))
    
    # Definir o system:index para que o toBands() o use como prefixo
    return img_selected.set('system:index', prefix)

def export_month(start_date_str, end_date_str, roi_ee):
    """Filtra, processa e exporta uma task para um mês específico."""
    print(f"Preparando exportação para o período: {start_date_str} a {end_date_str}")
    
    dataset = ee.ImageCollection(COLLECTION_ID) \
                .filterBounds(roi_ee) \
                .filterDate(start_date_str, end_date_str)
    
    # Verificar se há imagens no período
    count = dataset.size().getInfo()
    if count == 0:
        print(f"Sem dados para {start_date_str}. Pulando...")
        return

    # Aplicar a função de renomeação
    prepared_col = dataset.map(rename_and_prepare)
    
    # Transforma em Cubo (Bandas)
    data_cube = prepared_col.toBands()
    
    # Nome da Task
    clean_start = start_date_str.replace('-', '')
    task_name = f"ERA5_Cube_{clean_start}"
    
    try:
        task = ee.batch.Export.image.toDrive(
            image=data_cube,
            description=task_name,
            folder='MscDN_ERA5_Data',
            fileNamePrefix=f'era5_cube_{clean_start}',
            crs='EPSG:4326',
            scale=11131.949,
            region=roi_ee,
            maxPixels=1e13,
            fileFormat='GeoTIFF'
        )
        task.start()
        print(f"Task iniciada: {task_name} (ID: {task.id})")
    except Exception as e:
        print(f"Erro ao iniciar task {task_name}: {e}")

def main():
    # Inicializar GEE
    try:
        ee.Initialize(project='ee-lixoes')
    except Exception as e:
        print("Erro: Earth Engine não inicializado. Execute 'earthengine authenticate' no terminal.")
        ee.Authenticate() 
        ee.Initialize(project='ee-lixoes')
        return

    # 1. Carregar ROI
    if not os.path.exists(ROI_FILE):
        print(f"Erro: Arquivo ROI não encontrado em {ROI_FILE}")
        return
        
    gdf_roi = gpd.read_file(ROI_FILE)
    bounds = gdf_roi.total_bounds
    roi_ee = ee.Geometry.Rectangle([bounds[0], bounds[1], bounds[2], bounds[3]])

    print(f"--- Iniciando Pipeline ERA5 por Meses ---")

    # 2. Gerar lista de meses entre START e END
    current_date = datetime.strptime(START_DATE, '%Y-%m-%d')
    end_limit = datetime.strptime(END_DATE, '%Y-%m-%d')

    while current_date < end_limit:
        # Calcular o primeiro dia do próximo mês
        if current_date.month == 12:
            next_month = current_date.replace(year=current_date.year + 1, month=1, day=1)
        else:
            next_month = current_date.replace(month=current_date.month + 1, day=1)
        
        # Garantir que não ultrapasse a data final total
        actual_end = next_month if next_month < end_limit else end_limit
        
        start_str = current_date.strftime('%Y-%m-%d')
        end_str = actual_end.strftime('%Y-%m-%d')
        
        # Disparar Exportação
        export_month(start_str, end_str, roi_ee)
        
        # Avançar para o próximo mês
        current_date = next_month

    print(f"\n--- Todas as tasks foram enviadas! ---")
    print(f"Verifique o progresso no Task Manager: https://code.earthengine.google.com/tasks")

if __name__ == "__main__":
    main()

