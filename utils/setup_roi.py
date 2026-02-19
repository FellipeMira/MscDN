import os
import json
import geobr
import geopandas as gpd
from shapely.geometry import box

# --- Configurações ---
MUNICIPO_NOME = 'São Sebastião'
UF = 'SP'
BUFFER_DIST_KM = 50
OUTPUT_FILE = r'/home/mira/Desktop/MscDN/data/vector/roi_SP.geojson'

def main():
    print(f"Buscando geometria para {MUNICIPO_NOME} ({UF})...")
    
    # 1. Carregar município usando geobr
    # read_municipality retorna um GeoDataFrame
    try:
        mun = geobr.read_state(code_state=UF)
        #roi = mun[(mun['name_muni'] == MUNICIPO_NOME) & (mun['abbrev_state'] == UF)]
        roi = mun 
        if roi.empty:
            raise ValueError(f"Município {MUNICIPO_NOME}-{UF} não encontrado.")
    except Exception as e:
        print(f"Erro ao acessar geobr: {e}")
        return

    # 2. Assegurar CRS projetado para buffer em metros/km (SIRGAS 2000 / Brazil Polyconic ou UTM)
    # Vamos usar SIRGAS 2000 / UTM zone 23S (EPSG:31983) para SP, que é métrico.
    roi_proj = roi.to_crs(epsg=31983)
    
    # 3. Aplicar Buffer (100 km = 100,000 metros)
    print(f"Aplicando buffer de {BUFFER_DIST_KM} km...")
    roi_buffer = roi_proj.buffer(BUFFER_DIST_KM * 1000)
    
    # 4. Voltar para coordenadas geográficas (WGS84) para uso no Earth Engine
    roi_buffer_wgs84 = roi_buffer.to_crs(epsg=4326)
    
    # 5. Pegar o Bounding Box (Envelope) para simplificar a requisição no GEE
    # O GEE lida melhor com retângulos (bounds) do que geometrias complexas para exportação
    bbox = roi_buffer_wgs84.envelope
    
    # Criar GeoDataFrame final
    gdf_final = gpd.GeoDataFrame({'geometry': bbox}, crs='EPSG:4326')
    
    # 6. Salvar e mostrar infos
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
    
    gdf_final.to_file(OUTPUT_FILE, driver='GeoJSON')
    
    bounds = gdf_final.total_bounds
    area_km2 = roi_buffer.area.iloc[0] / 1e6
    
    print("\n--- Resumo do ROI ---")
    print(f"Arquivo salvo: {OUTPUT_FILE}")
    print(f"Área aproximada (com buffer): {area_km2:.2f} km²")
    print(f"Bounding Box (WGS84): {bounds}")

if __name__ == "__main__":
    main()