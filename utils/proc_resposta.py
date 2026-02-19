import os

# Desabilita o bloqueio de arquivos HDF5 para evitar erros em certos sistemas de arquivos
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import xarray as xr
import pandas as pd
import numpy as np
import glob
import os
import rioxarray  # Essencial para abrir GeoTIFFs no xarray
from dask.distributed import Client

def build_and_process_cube(input_dir, output_netcdf, p95_output):
    """
    Lê GeoTIFFs mensais, cria um cubo de dados virtual (Dask),
    calcula o P95 pixel-a-pixel e salva os resultados.
    """
    # 1. Configuração do Cluster Dask Local
    # Otimizado para máquinas locais com gerenciamento de memória
    client = Client(n_workers=2, threads_per_worker=2, memory_limit='4GB') 
    print(f"Cluster Dask iniciado: {client.dashboard_link}")
    
    try:
        # 2. Listagem de Arquivos
        search_path = os.path.join(input_dir, "ERA5_Land_Precip_*.tif")
        tif_files = sorted(glob.glob(search_path))
        
        if not tif_files:
            raise FileNotFoundError(f"Nenhum arquivo GeoTIFF encontrado em: {search_path}")

        # 3. Leitura Lazy e Concatenação
        datasets = []
        
        print(f"Montando cubo de dados para {len(tif_files)} arquivos...")
        for f in tif_files:
            # Abre o GeoTIFF usando rioxarray como backend
            # chunks={'x': 100, 'y': 100} fragmenta o espaço para paralelismo
            ds = xr.open_dataset(f, engine='rasterio', chunks={'x': 100, 'y': 100})
            
            # Extrair ano e mês do nome do arquivo
            filename = os.path.basename(f)
            parts = filename.replace('.tif', '').split('_')
            try:
                year, month = int(parts[-2]), int(parts[-1])
            except (ValueError, IndexError):
                print(f"Aviso: Não foi possível extrair data do arquivo {filename}. Pulando.")
                continue
            
            # Ajustar dimensões: GeoTIFF lido pelo rioxarray tem 'band', 'x', 'y'
            # Precisamos converter 'band' para 'time'
            num_bands = ds.band.size
            time_index = pd.date_range(
                start=f"{year}-{month:02d}-01", 
                periods=num_bands, 
                freq='h' # 'h' para hourly (ERA5-Land é horário)
            )
            
            ds = ds.rename({'band': 'time'})
            ds = ds.assign_coords(time=time_index)
            
            # Garante que as variáveis tenham nomes coerentes
            if 'band_data' in ds.data_vars:
                da = ds['band_data'].rename('precipitation')
            else:
                # Caso o rasterio/rioxarray use outro nome padrão
                primary_var = list(ds.data_vars)[0]
                da = ds[primary_var].rename('precipitation')
                
            datasets.append(da)

        # Concatena todos os meses ao longo do tempo
        full_ds = xr.concat(datasets, dim='time', join='override', coords='minimal')
        
        # PADRONIZAÇÃO DE COORDENADAS PARA GIS (QGIS/ArcGIS)
        # Renomeia x/y para longitude/latitude para melhor compatibilidade
        if 'x' in full_ds.dims and 'y' in full_ds.dims:
            full_ds = full_ds.rename({'x': 'longitude', 'y': 'latitude'})
        
        # Define atributos padrões CF (Climate and Forecast)
        full_ds.longitude.attrs = {'units': 'degrees_east', 'standard_name': 'longitude', 'long_name': 'longitude'}
        full_ds.latitude.attrs = {'units': 'degrees_north', 'standard_name': 'latitude', 'long_name': 'latitude'}
        full_ds.attrs['title'] = 'Cubo de Precipitacao ERA5-Land'
        full_ds.attrs['history'] = f'Gerado em {pd.Timestamp.now()}'
        
        # GARANTIR CRS: Importante para o QGIS reconhecer a localização
        full_ds.rio.write_crs("EPSG:4326", inplace=True)
        
        # Salvar o cubo principal
        # Usamos engine='h5netcdf' que é mais estável com Dask para evitar erros HDF
        print(f"Salvando cubo de dados completo em {output_netcdf}...")
        if os.path.exists(output_netcdf):
            os.remove(output_netcdf)
        full_ds.to_netcdf(output_netcdf, engine='h5netcdf')
        
        # 4. Cálculo do Percentil 95 (P95)
        # DATA LEAKAGE PREVENTION: Usar apenas o período de treino
        train_end_date = '2025-12-31' 
        
        # Filtra os dados de treino
        ds_train = full_ds.sel(time=slice(None, train_end_date))
        
        # Se o filtro resultar em vazio (ex: dados apenas de 2025), usa o que estiver disponível
        # mas avisa o usuário.
        if ds_train.time.size == 0:
            print(f"Aviso: Nenhum dado encontrado antes de {train_end_date}. Usando todo o dataset.")
            ds_train = full_ds
        else:
            print(f"Calculando P95 usando dados de treino até {train_end_date} ({ds_train.time.size} timesteps)...")

        # OTIMIZAÇÃO CRÍTICA PARA QUANTILE:
        # Rechunk para ter o tempo contíguo em cada bloco espacial.
        print("Otimizando chunks para cálculo de estatísticas...")
        ds_train = ds_train.chunk({'time': -1, 'latitude': 50, 'longitude': 50})
        
        # Executa o cálculo
        print("Processando P95 (isso pode levar alguns minutos)...")
        p95_map = ds_train.quantile(0.95, dim='time', skipna=True).compute()
        
        # Garante que o CRS e atributos sejam mantidos no mapa P95
        p95_map.rio.write_crs("EPSG:4326", inplace=True)
        p95_map.attrs['units'] = 'mm'
        p95_map.attrs['long_name'] = '95th percentile of precipitation'
        
        # Salva o resultado em formato TIFF (mais amigável para QGIS/GIS)
        print(f"Salvando mapa de P95 em {p95_output}...")
        if os.path.exists(p95_output):
            os.remove(p95_output)
            
        # Para exportar TIFF, o rioxarray prefere dimensões nomeadas x e y
        if 'longitude' in p95_map.dims:
            p95_export = p95_map.rename({'longitude': 'x', 'latitude': 'y'})
        else:
            p95_export = p95_map
            
        p95_export.rio.to_raster(p95_output)
        print(f"Sucesso! Mapa de P95 salvo em {p95_output}")
        
        return full_ds, p95_map

    finally:
        if 'client' in locals():
            client.close()

# Execução
if __name__ == "__main__":
    # Caminhos baseados na estrutura do projeto
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT_DIR = os.path.join(BASE_DIR, "data", "raster", "resposta")
    OUTPUT_CUBE = os.path.join(BASE_DIR, "data", "raster", "cubo_precipitacao.nc")
    OUTPUT_P95 = os.path.join(BASE_DIR, "data", "raster", "p95_precipitacao.tif")

    # Verifica se o diretório de entrada existe
    if os.path.exists(INPUT_DIR):
        ds, p95 = build_and_process_cube(INPUT_DIR, OUTPUT_CUBE, OUTPUT_P95)
    else:
        print(f"Erro: Diretório de entrada não encontrado: {INPUT_DIR}")
