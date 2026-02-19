import ee
import geemap
import geopandas as gpd
from datetime import datetime

# Autenticação e Inicialização do GEE
try:
    ee.Initialize(project='ee-lixoes')
except Exception as e:
    ee.Authenticate() 
    ee.Initialize(project='ee-lixoes')

ROI_FILE = r'data/vector/ValeDoParaiba.geojson'
gdf_roi = gpd.read_file(ROI_FILE)
bounds = gdf_roi.total_bounds
ROI = ee.Geometry.Rectangle([bounds[0], bounds[1], bounds[2], bounds[3]])

START_YEAR = 2025 #1995
END_YEAR = 2025
VAR_BAND = 'total_precipitation' 
FOLDER_DRIVE = "ERA5_Land_Hourly_Precipitation"

def preprocess_image(image):
    """
    Função mapeada sobre a coleção para pré-processamento.
    1. Seleciona a banda de precipitação.
    2. Converte de Metros (m) para Milímetros (mm).
    3. Renomeia para evitar conflitos na exportação.
    """
    # Multiplica por 1000 para converter m -> mm
    precip_mm = image.select(VAR_BAND).multiply(1000).toFloat()
    
    # Formata a data para ser usada no nome da banda (crucial para toBands)
    date_string = image.date().format("yyyyMMdd_HH")
    
    # Renomeia a banda usando a data. 
    # Obs:.toBands() prefixa automaticamente o ID da imagem, mas renomear ajuda na clareza.
    return precip_mm.rename(ee.String("precip_mm_").cat(date_string)) \
                   .set('system:time_start', image.get('system:time_start'))

def export_data_batch(roi, start_year, end_year, folder):
    """
    Gerencia a exportação em lotes mensais para contornar o limite de 5000 bandas
    do método.toBands() e limites de memória do GEE.
    """
    collection = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
    
    tasks = []
    
    print(f"Iniciando agendamento de tarefas de {start_year} a {end_year}...")
    
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # Define janela temporal do mês
            start_date = ee.Date.fromYMD(year, month, 1)
            # Avança 1 mês para o fim do intervalo
            end_date = start_date.advance(1, 'month')
            
            # Filtra a coleção
            # filterBounds reduz o processamento apenas à área relevante antes do download
            filtered_col = collection.filterDate(start_date, end_date) \
                                    .filterBounds(roi) \
                                    .map(preprocess_image)
            
            # Checagem de segurança (opcional): verificar se há imagens
            # count = filtered_col.size().getInfo() (evitar em loops grandes se possível)
            
            # --- USO DO.toBands() ---
            # Converte a ImageCollection (T imagens de 1 banda) em 1 Imagem de T bandas.
            # Um mês tem ~720 a 744 horas, o que é seguro (< 5000).
            stacked_image = filtered_col.toBands()
            
            # Definição do nome do arquivo
            # Ex: ERA5_Land_Precip_1994_01
            filename = f"ERA5_Land_Precip_{year}_{month:02d}"
            
            # Configuração da Tarefa de Exportação
            # Usamos a resolução e alinhamento nativos do ERA5-Land (0.1 deg)
            task = ee.batch.Export.image.toDrive(
                image=stacked_image,
                description=filename,
                folder=folder,
                fileNamePrefix=filename,
                region=roi,
                crs='EPSG:4326',
                scale=11131.949,
                maxPixels=1e13,
                fileFormat='GeoTIFF' 
            )
            
            # Inicia a tarefa (assíncrono)
            task.start()
            tasks.append(f"{filename}: {task.status()['state']}")
            
    return tasks

# Execução principal
if __name__ == "__main__":
    task_list = export_data_batch(ROI, START_YEAR, END_YEAR, FOLDER_DRIVE)
    print(f"Total de tarefas submetidas: {len(task_list)}")
    print("Verifique o status na aba 'Tasks' do Code Editor ou via task.status()")