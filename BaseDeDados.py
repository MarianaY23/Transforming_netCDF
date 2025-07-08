# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 13:54:01 2025


Precisamos criar um arquivo em netCDF com os dados de monitoramento da qualida-
de do ar. O arquivo terá as seguintes dimensões: 
    Poluente
    Tempo
    Latitude
    Longitude
Ou seja, aqData(Poluente,Tempo,Latitude,Longitude).
Faremos um buffer de 5 km ao redor da estação de monitoramento e verificaremos 
qual o cobertura do pixel que esta estação representa.


@author: marit
"""
#%%

import geopandas as gpd
import pandas as pd
import numpy as np    
from shapely.ops import polygonize
from shapely.geometry import MultiLineString
import matplotlib.pyplot as plt
from shapely.geometry import Point
import os
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
from shapely.geometry import mapping


#%% Dados que serão utilizados e já tratados 
df_column = pd.read_csv(r"C:\PYTHON\ENS5132\ENS5132\Trabalho_03\outputs\estacoesInput.csv")
df_pol_SP = pd.read_csv(r"C:\PYTHON\ENS5132\ENS5132\Trabalho_03\outputs\df_pol_SP.csv")
#%%
def gridding(lon,lat):
    """
    Esta função cria uma grade regular de polígonos a partir de vetores de 
    latitude e longitude, com seus respectivos centróides e pontos da malha,
    ideal para análises geoespaciais, interpolação ou visualização.

    Parameters
    ----------
    lon : TYPE
        DESCRIPTION.
    lat : TYPE
        DESCRIPTION.

    Returns
    -------
    grids : TYPE
        DESCRIPTION.
    xv : TYPE
        coordenada x de cada ponto da malha
    yv : TYPE
        coordenada y de cada ponto da malha
    xX : TYPE
        coordenada x do centroide da célula de grade.
    yY : TYPE
        coordenada y do centroide da célula de grade.

    """
    print('Calling gridding function')
    xv, yv = np.meshgrid(lon, lat)
    hlines = [((x1, yi), (x2, yi)) for x1, x2 in zip(lon[:-1], lon[1:]) for yi in lat]
    vlines = [((xi, y1), (xi, y2)) for y1, y2 in zip(lat[:-1], lat[1:]) for xi in lon]
    grids = list(polygonize(MultiLineString(hlines + vlines)))
    grids = gpd.GeoDataFrame(grids) 
    grids.columns =['geometry'] 
    grids['geometry'] = grids['geometry']
    grids = grids.set_geometry(grids['geometry'])
    grids.crs = "EPSG:4326"  
    grids['X'] = grids.geometry.centroid.x
    grids['Y'] = grids.geometry.centroid.y
    xX = np.array(grids['X']).reshape((lon.shape[0]-1,lat.shape[0]-1)).transpose()
    yY = np.array(grids['Y']).reshape((lon.shape[0]-1,lat.shape[0]-1)).transpose()
    return grids,xv,yv,xX,yY




#%%

def populatingGrid(st,xX,yY,pollutants,DateStart,DateEnd):  
    """
    Preenche a grade espacial com dados de poluentes.

    Parâmetros
    ----------
    st : DataFrame
        Estações de monitoramento com colunas 'lon', 'lat', 'Poluentes', 'Codigo'.
    xX : ndarray
        Matriz de longitudes dos centróides da grade.
    yY : ndarray
        Matriz de latitudes dos centróides da grade.
    pollutants : list
        Lista de poluentes a processar.
    DateStart : str
        Data inicial no formato 'YYYY-MM-DD HH:MM'.
    DateEnd : str
        Data final no formato 'YYYY-MM-DD HH:MM'.
    Returns
    -------
    data : TYPE
        DESCRIPTION.

    """
    print('Calling populatingGrid function')
    

    # Criando matriz de dados
   

    
    for jj, pol in enumerate(pollutants):
        data = np.zeros([st.shape[0],dates.shape[0],np.size(yY,0), np.size(xX,1)])
        data[data==0] = np.nan
        
        for ii in range(0,st.shape[0]):
            
            if pol in st.Poluentes.str.split(';').replace(' ','')[ii]:
                stPath = "C:\PYTHON\ENS5132\ENS5132\Trabalho_03\outputs" + "/"+pol+"/"+st.Codigo[ii]+"_"+pol+".csv"
               
                if os.path.exists(stPath):
                    print(st['Codigo'][ii] + ' verdade')
                    print(stPath)
                    df = pd.read_csv(stPath)
                    df['datetime'] = pd.to_datetime( df['datetime'])
                    merged_df = pd.merge(df, dates, left_on='datetime', right_on='datetime')
                    # 
                    dist = ((st.lon[ii]-xX)**2 + (st.lat[ii]-yY)**2)**(1/2)
                    mindist = np.where(dist == np.amin(dist))
                    print('st number '+str(ii)+' from '+str(st.shape[0]))
                    data[ii,:,mindist[0],mindist[1]]= np.array(merged_df.Valor)
        data = np.nanmean(data,axis=0)
        if jj ==0:
            dataNew = data.copy()
        else:
            dataNew = np.concatenate([dataNew,data],-1)
            # Criar DataArray xarray

    return dataNew    

    data_xr = xr.DataArray(
    data,
        dims=("tempo", "lat", "lon"),
        coords={
            "tempo": dates["datetime"],
            "lat": yY[:,0],    # Ajuste dependendo do formato do yY
            "lon": xX[0,:],    # Ajuste dependendo do formato do xX
            #"poluente": pollutants
        },
        name="concentracao"
    )
    
    # Salvar em NetCDF
   # arquivo_nc = (r"C:\PYTHON\ENS5132\ENS5132\Trabalho_03\outputs\MP10_SP_monitoramento.nc")
   # data_xr.to_netcdf(arquivo_nc)
    #print(f"Arquivo NetCDF salvo em: {arquivo_nc}")            
    
# fazer uma planilha pra cada estacao que junte na coluna de poluente os poluentes que ela mede MP10; NO3... 

#%% criando dados 
# fazer uma planilha pra cada estacao que junte na coluna de poluente os poluentes que ela mede MP10; NO3... 
gdfEstados = gpd.read_file(r"C:\PYTHON\ENS5132\ENS5132\Trabalho_03\inputs\SP_UF_2024\SP_UF_2024.shp")

# Selecionar SP
gdf_SP = gdfEstados[gdfEstados.NM_UF =='São Paulo']
gdf_SP.plot()

# Extraindo bounds de SP
boundsSP = gdf_SP.bounds

# extraindo coordenadas mais externas do domínio
lowerLat = np.min(boundsSP.miny)
upperLat = np.max(boundsSP.maxy)
leftLon = np.min(boundsSP.minx)
rightLon = np.max(boundsSP.maxx)

# definir espaçamento da grade - USUÁRIO DEFINE
dx = 0.1
dy = 0.1
#%% MAIN

# criando vetor de latitudes e longitudes
lat = np.arange(lowerLat-dy,upperLat+dy,dy)
lon = np.arange(leftLon-dx,rightLon+dx,dx) 


grids,xv,yv,xX,yY=gridding(lon,lat)   


st = pd.read_csv(r"C:\PYTHON\ENS5132\ENS5132\Trabalho_03\outputs\estacoesInput.csv")

pollutants = ['MP10']
st = st.dropna(axis='rows')
DateStart = '2023-01-01 00:00'
DateEnd = '2023-12-31 23:00'
    
# Criando vetor de datas       
dates = pd.date_range(start=DateStart, end=DateEnd, freq = 'h')
dates = pd.DataFrame({"datetime":pd.to_datetime(dates)})
    
    
    
data = populatingGrid(st,xX,yY,pollutants,DateStart,DateEnd)  
    

# Teste
fig, ax = plt.subplots()
ax.pcolor(xX,yY,data[0,0,:,:])
gdf_SP.boundary.plot(ax=ax)


#%% Fazendo a média anual de cada pixel

# === 1. Abrir NetCDF ===
ds = xr.open_dataset(r'C:\PYTHON\ENS5132\ENS5132\Trabalho_03\outputs\MP10_SP_monitoramento.nc')

print(ds)  

# === 2. Calcular a média anual ===
# Se tiver dados de vários anos
media_anual = ds.concentracao.groupby('tempo.year').mean(dim='tempo')

# Se for só um ano no dataset (ou quiser a média total):
# media_anual = ds.concentracao.mean(dim='tempo')

print(media_anual)

# === 3. Carregar o shapefile do Estado de SP ===
gdf = gpd.read_file(r"C:\PYTHON\ENS5132\ENS5132\Trabalho_03\inputs\SP_UF_2024\SP_UF_2024.shp")

# Se necessário, reprojeta para WGS84 (graus decimais)
if gdf.crs != "EPSG:4326":
    gdf = gdf.to_crs("EPSG:4326")

# === 4. Plotar a média anual ===
# Seleciona o ano desejado (exemplo: 2023)
media_plot = media_anual.sel(year=2023)

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 8))

media_plot.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='viridis', cbar_kwargs={"label": "Média Anual da Concentração CO"})

# Adiciona o contorno do estado
gdf.boundary.plot(ax=ax, color='red', linewidth=1)

ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')


ax.set_title('Média Anual da Concentração MP10 (Ano 2023)')
ax.coastlines()
plt.show()

# salvar a imagem 
# === 5. Salvar a média anual em um novo arquivo NetCDF ===
fig.savefig(r"C:\PYTHON\ENS5132\ENS5132\Trabalho_03\outputs\MediaAnual_CO_SP") 
print("Média anual salva com sucesso!")

#%% Média por estações do ano 

def get_estacao(mes):
    """
    Essa função faz a divisão das estações do ano (primavera, verão, outono e inverno)

    Parameters
    ----------
    mes : TYPE
        DESCRIPTION.

    Returns
    -------
    str
        DESCRIPTION.

    """
    if mes in [12, 1, 2]:
        return 'Verão'
    elif mes in [3, 4, 5]:
        return 'Outono'
    elif mes in [6, 7, 8]:
        return 'Inverno'
    else:
        return 'Primavera'
# Aplicar função para criar estação do ano
ds = ds.assign_coords(estacao_ano=('tempo', [get_estacao(pd.Timestamp(d).month) for d in ds['tempo'].values]))

# Calcular média por estação do ano
media_estacoes = ds['concentracao'].groupby('estacao_ano').mean(dim='tempo')

print(media_estacoes)

# === 4. Ler shapefile do limite de São Paulo ===
gdf = gpd.read_file(r"C:\PYTHON\ENS5132\ENS5132\Trabalho_03\inputs\SP_UF_2024\SP_UF_2024.shp")

# === 5. Plotar cada estação ===
projecao = ccrs.PlateCarree()
for estacao in media_estacoes['estacao_ano'].values:
    fig, ax = plt.subplots(subplot_kw={'projection': projecao}, figsize=(10, 8))

    # Plotar o mapa com média da estação
    media_estacoes.sel(estacao_ano=estacao).plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='viridis',
        cbar_kwargs={"label": f"Média da Concentração - {estacao}"}
    )

    # Adicionar limite de SP
    gdf.to_crs("EPSG:4326").boundary.plot(ax=ax, edgecolor='red', linewidth=1.5)

    # Configurar mapa
    ax.set_title(f"Concentração Média - {estacao}")
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.gridlines(draw_labels=True)

 #   plt.savefig(fr'C:\PYTHON\ENS5132\ENS5132\Trabalho_03\outputs\media_{estacao}_SP.png', dpi=300, bbox_inches='tight')
  #  plt.close()

print("Gráficos salvos com sucesso!")

#%% recortando o netCDF para o mnicipio de SP 

# 1. Ler shapefile dos municípios de SP
cities_path = r"C:\PYTHON\ENS5132\ENS5132\Trabalho_03\inputs\SP_Municipios_2024\SP_Municipios_2024.shp"
cities = gpd.read_file(cities_path)

# 2. Selecionar a cidade desejada (São Paulo)
cidade = cities[cities['NM_MUN'] == 'São Paulo']

# 3. Ler o NetCDF
nc_path = r'C:\PYTHON\ENS5132\ENS5132\Trabalho_03\outputs\MP10_SP_monitoramento.nc'
ds = xr.open_dataset(nc_path)

# 4. Configurar dimensões espaciais no rioxarray
ds = ds.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
ds = ds.rio.write_crs("EPSG:4326")

# 5. Preparar geometria para recorte
geoms = [mapping(geom) for geom in cidade.geometry]

# 6. Recortar o NetCDF para a cidade
recorte = ds.rio.clip(geoms, cidade.crs, drop=True)

# 7. Salvar recorte (opcional)
recorte_path = r'C:\PYTHON\ENS5132\ENS5132\Trabalho_03\outputs\recorte_SaoPaulo.nc'
recorte.to_netcdf(recorte_path)
print("✅ Recorte concluído e salvo!")

# 8. Abrir recorte para análise
recorte = xr.open_dataset(recorte_path)
dados = recorte['concentracao']

# ----- FUNÇÕES DE PLOT ----- #

def plot_media_anual(dados, cidade):
    media_anual = dados.mean(dim='tempo')
    fig, ax = plt.subplots(figsize=(10, 8))
    media_anual.plot(cmap='viridis', ax=ax)
    cidade.boundary.plot(ax=ax, edgecolor='black', linewidth=2)
    minx, miny, maxx, maxy = cidade.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_title("Concentração Média Anual - São Paulo")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.show()


def plot_serie_temporal_media(dados):
    media_espacial_tempo = dados.mean(dim=['lat', 'lon'])
    plt.figure(figsize=(12, 5))
    media_espacial_tempo.plot()
    plt.title("Média Espacial da Concentração ao longo do tempo - São Paulo")
    plt.xlabel("Tempo")
    plt.ylabel("Concentração Média")
    plt.grid(True)
    plt.show()

# ----- EXECUTANDO OS PLOTS ----- #

plot_media_anual(dados, cidade)
#plt.savefig(fr'C:\PYTHON\ENS5132\ENS5132\Trabalho_03\outputs\cidade_SP_médiaAnual.png', dpi=300, bbox_inches='tight')
  #  plt.close()
  
plot_serie_temporal_media(dados)
#plt.savefig(fr'C:\PYTHON\ENS5132\ENS5132\Trabalho_03\outputs\SP_cidade_pixels_estacoes.png', dpi=300, bbox_inches='tight')
  #  plt.close()


#%% Plotando todas as cidades com as concentrações médias por cidade a partir da média das estações que medem alguma área da cidade


# 1. Ler shapefile do estado de SP (limite)
estado_path = r"C:\PYTHON\ENS5132\ENS5132\Trabalho_03\inputs\SP_UF_2024\SP_UF_2024.shp"
estado_sp = gpd.read_file(estado_path)

# 2. Ler shapefile dos municípios de SP
cities_path = r"C:\PYTHON\ENS5132\ENS5132\Trabalho_03\inputs\SP_Municipios_2024\SP_Municipios_2024.shp"
cities = gpd.read_file(cities_path)

# 3. Ler NetCDF e configurar spatial dims
nc_path = r'C:\PYTHON\ENS5132\ENS5132\Trabalho_03\outputs\MP10_SP_monitoramento.nc'
ds = xr.open_dataset(nc_path)
ds = ds.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
ds = ds.rio.write_crs("EPSG:4326")

# 4. Extrair coordenadas dos pixels
lons = ds['lon'].values
lats = ds['lat'].values
lon_grid, lat_grid = np.meshgrid(lons, lats)
points = [Point(x, y) for x, y in zip(lon_grid.flatten(), lat_grid.flatten())]
gdf_pixels = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")

# 5. Spatial join: associar pixels a municípios
pixels_cities = gpd.sjoin(gdf_pixels, cities, how="left", predicate='within')

# 6. Converter dados concentração para DataFrame (pixels x tempo)
concentracao = ds['concentracao'].values  # (tempo, lat, lon)
tempo = ds['tempo'].values
concentracao_flat = concentracao.reshape((concentracao.shape[0], -1))  # (tempo, pixels)
df_conc = pd.DataFrame(concentracao_flat.T, columns=tempo)  # pixels x tempo
df_conc['municipio'] = pixels_cities['NM_MUN'].values

# 7. Calcular média por município para cada tempo
df_mun = df_conc.groupby('municipio').mean()

# 8. Calcular média anual
df_mun['media_anual'] = df_mun.mean(axis=1)

# 9. Merge para GeoDataFrame municípios com média anual
cities_mun = cities.merge(df_mun['media_anual'], left_on='NM_MUN', right_index=True)

# Filtrar apenas municípios com dados
cities_mun_com_dados = cities_mun.dropna(subset=['media_anual'])

# --- PLOT COM LEGENDA EM TABELA E GRADE --- #

# Ordenar os municípios por concentração (opcional, deixa a tabela mais organizada)
cities_mun_com_dados = cities_mun_com_dados.sort_values(by='media_anual', ascending=False)

fig = plt.figure(figsize=(20, 16))
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])  # mapa mais largo que a tabela

# --- Mapa ---
ax0 = plt.subplot(gs[0])
cities_mun_com_dados.plot(column='media_anual', cmap='viridis', legend=True, edgecolor='black', ax=ax0)
estado_sp.boundary.plot(ax=ax0, edgecolor='red', linewidth=2)
cities.boundary.plot(ax=ax0, edgecolor='black', linewidth=0.5)

# Adicionando grade (linhas de latitude/longitude)
ax0.set_xticks(np.arange(-53.5, -44, 1))  # Ajuste conforme a área do estado de SP
ax0.set_yticks(np.arange(-26, -19, 1))
ax0.grid(True, which='both', color='gray', linestyle='--', linewidth=0.3)

# Rótulos dos eixos
ax0.set_xlabel('Longitude')
ax0.set_ylabel('Latitude')

ax0.set_title("Média Anual da Concentração de MP10 por Município - SP ", fontsize=16)

# --- Tabela ---
ax1 = plt.subplot(gs[1])
ax1.axis('off')
table_data = list(zip(cities_mun_com_dados['NM_MUN'], cities_mun_com_dados['media_anual'].round(2)))
table = ax1.table(cellText=table_data, colLabels=['Município', 'MP10 (Média Anual)'],
                  loc='center', cellLoc='center', colLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)


# salvando a figura 
plt.savefig(r"C:\PYTHON\ENS5132\ENS5132\Trabalho_03\figures\SP_municipios.png")

plt.show()




#%% HISTOGRAMA
import matplotlib.pyplot as plt

municipio_escolhido = 'São Paulo'  # coloque o município que quiser

# Pegar todas as concentrações dos pixels daquele município
df_municipio = df_conc[df_conc['municipio'] == municipio_escolhido].drop(columns='municipio')

# "Empilhar" os dados para 1D (todos os valores concentrados)
valores = df_municipio.values.flatten()

plt.figure(figsize=(8,5))
plt.hist(valores, bins=30, color='skyblue', edgecolor='black')
plt.title(f"Histograma da concentração de MP10 - {municipio_escolhido}")
plt.xlabel("Concentração")
plt.ylabel("Frequência")
plt.show()

#%% Percentil 90 comparado com a CONAMA apenas cidade de SP 
# MP10 

# === 1. Abrir o NetCDF (já recortado para SP) ===
nc_path = r'C:\PYTHON\ENS5132\ENS5132\Trabalho_03\outputs\MP10_SP_monitoramento.nc'
ds = xr.open_dataset(nc_path)

# === 2. Calcular a média espacial ===
media_espacial = ds['concentracao'].mean(dim=['lat', 'lon'])  # Agora só varia no tempo

# === 3. Converter para pandas e calcular média diária ===
df = media_espacial.to_dataframe().reset_index()
df['tempo'] = pd.to_datetime(df['tempo'])
media_diaria = df.groupby(df['tempo'].dt.date)['concentracao'].mean()

# === 4. Calcular percentil 90 e dias acima do limite ===
percentil_90 = np.percentile(media_diaria, 90)
dias_acima_limite = (media_diaria > 45).sum()

# === 5. Exibir resultados ===
print("Percentil 90% da média diária:", round(percentil_90, 2))
print("Dias acima do limite (45 µg/m³):", dias_acima_limite)

# === 6. Gráfico ===
plt.figure(figsize=(14, 6))
plt.plot(media_diaria.index, media_diaria.values, label='Média Diária MP10', color='blue')
plt.axhline(y=percentil_90, color='green', linestyle='--', linewidth=2,
            label=f'Percentil 90% ({percentil_90:.2f} µg/m³)')
plt.axhline(y=45, color='red', linestyle='--', linewidth=2, label='Limite 45 µg/m³')

plt.title('Média Diária de MP10 na Cidade de São Paulo - 2023')
plt.xlabel('Data')
plt.ylabel('Concentração MP10 (µg/m³)')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

plt.text(0.02, 0.95,
         f'Dias acima do limite (45 µg/m³): {dias_acima_limite}',
         transform=plt.gca().transAxes,
         fontsize=12,
         verticalalignment='top',
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

# Salvar o gráfico (opcional)
plt.savefig(r'C:\PYTHON\ENS5132\ENS5132\Trabalho_03\figures\grafico_mp10_sp.png', dpi=300, bbox_inches='tight')

plt.show()

#%%

import xarray as xr
import matplotlib.pyplot as plt
import os

# Abrir NetCDF
nc_path = r'C:\PYTHON\ENS5132\ENS5132\Trabalho_03\outputs\MP10_SP_monitoramento.nc'
ds = xr.open_dataset(nc_path)

# Calcular média anual (média espacial + temporal)
media_anual = ds['concentracao'].mean(dim=['lat', 'lon', 'tempo']).item()

# Limite anual (exemplo)
limite_anual = 15

# Dados para o gráfico
valores = [media_anual, limite_anual]
labels = ['Média Anual MP10', 'Limite Anual']

# Criar pasta caso não exista
pasta_figures = r"C:\PYTHON\ENS5132\ENS5132\Trabalho_03\figures"
os.makedirs(pasta_figures, exist_ok=True)

# Plot
plt.figure(figsize=(6, 6))
bars = plt.bar(labels, valores, color=['blue', 'red'])
plt.ylabel('Concentração (µg/m³)')
plt.title('Média Anual MP10 x Limite Anual (São Paulo 2023)')

# Adicionar valores acima das barras
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.2f}', ha='center', fontsize=12)

plt.ylim(0, max(valores)*1.3)

# Salvar figura
plt.savefig(os.path.join(pasta_figures, "media_anual_SP_CONAMA.png"))
plt.show()
