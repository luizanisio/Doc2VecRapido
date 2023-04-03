# -*- coding: utf-8 -*-
#######################################################################
# Código complementar ao Doc2VecRapido para agrupar documentos
# Esse código, dicas de uso e outras informações: 
#   -> https://github.com/luizanisio/Doc2VecRapido/
# Luiz Anísio 
# 04/03/2023 - disponibilizado no GitHub  
#######################################################################

from util_doc2util import UtilDocs
import os 

import numpy as np
import pandas as pd
from datetime import datetime
import random
import json
from scipy import spatial
import seaborn as sns
from matplotlib import pyplot as plt
from collections import Counter
from sklearn.manifold import TSNE
from util_pandas import UtilPandasExcel

CST_TAMANHO_COLUNA_TEXTO = 250

''' 
> agrupar uma lista de vetores
  > retorna um objeto de agrupamento com um dataframe com os dados do agrupamento: grupo,centroide,similaridade,vetor
  grupos = AgrupamentoRapido.agrupar_vetores(vetores, sim)
  print(grupos.dados)

> agrupar arquivos de uma pasta gerando um excel no final
  > será gerado um aquivo "agrupamento {pasta de textos} sim {similaridade}.xlsx"
  > também retorna o objeto de agrupamento com o dataframe de agrupamento
  > arquivo_saida = None só retorna o datagrame em gerar o arquivo
  > se plotar = True, então vai gerar um arquivo "arquivo_saida.png"
  grupos = AgrupamentoRapido.agrupar_arquivos(pasta_modelo, pasta_arquivos, 
                                                      arquivo_saida = '', 
                                                      similaridade = 90,
                                                      plotar = True):
  print(grupos.dados)

> usar o objeto
  grupos = UtilAgrupamentoFacil(dados=meu_dataframe, similaridade=90)
  print(grupos.dados)
'''
def modelo_generico(modelo = '', testar = False):
    arq_bert = os.path.join(modelo, 'pytorch_model.bin')
    if  modelo in ('bert','bert-large','bert-base','bert_large','bert_base') or os.path.isfile(arq_bert):
       if testar: return True
       from util_doc2bert_rapido import Doc2BertRapido
       return Doc2BertRapido(modelo)
    # vazio é considerado o "meu_modelo"
    if (not modelo) or os.path.isdir(modelo):
       if testar: return True
       from util_doc2vec_rapido import Doc2VecRapido
       return Doc2VecRapido(modelo)
    # não é pasta local ou de rede, busca o modelo bert remoto  
    if modelo.find('.') < 0 and modelo.find(':') < 0 \
       and modelo.find('//') < 0 and modelo.find('\\') < 0:
       if testar: return True 
       from util_doc2bert_rapido import Doc2BertRapido
       return Doc2BertRapido(modelo)
    msg = f'Não foi possível identificar um modelo Doc2VecRapido ou Bert local ou remoto, use o nome da pasta ou bert-large ou bert-base'  
    raise Exception(msg)  

class AgrupamentoRapido():
    BATCH_VETORIZACAO = 1000

    # dados pode ser:
    # - uma lista de vetores
    # - um dict com vetores e metadados
    # - um dict com textos e metadados
    # - um dict com textos, vetores e metadados
    def __init__(self, dados = None, 
                       pasta_arquivos = None, 
                       pasta_modelo = None, 
                       epocas_vetorizacao = 100,
                       similaridade = 90, 
                       distancia = 'cosine',
                       arquivo_cache_vetores = None,
                       apenas_atualizar_cache = False,
                       arquivo_saida_excel = None,
                       coluna_texto = None,
                       plotar = False):
        if dados is not None:
            if type(dados) in (list, np.array, np.ndarray):
               if len(dados) > 0 and type(dados[0]) is not dict:
                  # recebeu uma lista de vetores
                  dados = [{'vetor': v} for v in dados]
        else:
            dados = [] 
        self.epocas_vetorizacao = epocas_vetorizacao
        self.coluna_texto = bool(coluna_texto)
        
        self.arquivo_cache_vetores = None
        # configuração de cache de vetores 
        if arquivo_cache_vetores == True:
           self.arquivo_cache_vetores = './__cache_vetores__.json'
        elif arquivo_cache_vetores: 
           self.arquivo_cache_vetores = arquivo_cache_vetores
        
        self.cache_vetores = {}
        # carrega os arquivos para o dicionário de dados se foi passada a pasta de arquivos
        dados_arquivos = self.carregar_arquivos_para_dados(pasta_arquivos)
        dados = dados + dados_arquivos
        self.vetorizar_textos_dados(dados, pasta_modelo=pasta_modelo)
        if apenas_atualizar_cache:
            UtilDocs.printlog('AgrupamentoRapido (atualizar cache)', f'Cache gerados para {len(dados)} textos')
            self.dados = None
            return 

        # dados do objeto é um dataframe dos dados de entrada
        self.dados = pd.DataFrame(dados)
        
        # finaliza para agrupamento            
        self.similaridade = similaridade if similaridade>1 else int(similaridade*100)
        self.distancia = 'cosine' if distancia.lower() in ('c','cosine') else 'euclidean'
        self.dados['vetor_np'] = [np.array(v) for v in self.dados['vetor']]
        self.dados['grupo'] = [-1 if v else 1 for v in self.dados['vetor']] # sem vetor é o grupo 1 (vazios)
        self.dados['centroide'] = [0 for _ in range(len(self.dados))]
        self.agrupar()
        if arquivo_saida_excel:
           self.to_excel(arquivo_saida_excel, self.similaridade)

        if plotar and arquivo_saida_excel:
           self.plotar(show_plot=False, arquivo_saida = arquivo_saida_excel, similaridade=similaridade)

    def vec_similaridades(self, vetor, lst_vetores):
        #_v = np.array(vetor) if type(vetor) is list else vetor
        _v = vetor.reshape(1, -1)
        return ( 1-spatial.distance.cdist(lst_vetores, _v, self.distancia).reshape(-1) )  
  
    def grupos_vetores(self):
        grupos = self.dados[self.dados.centroide == 1]
        vetores = list(grupos['vetor_np'])
        grupos = list(grupos['grupo'])
        return grupos, vetores
  
    def melhor_grupo(self, vetor):
      # busca os grupos e vetores dos centróides
      grupos, vetores = self.grupos_vetores()
      # retorna -1 se não existirem centróides
      if (not vetores):
          return -1, 0
      # busca a similaridade com os centróides
      sims = list(self.vec_similaridades(vetor,vetores))
      # verifica a maior similaridade
      maxs = max(sims)
      # busca o centróide com maior similaridade
      imaxs = sims.index(maxs) if maxs*100 >= self.similaridade else -1
      # retorna o número do grupo e a similaridade com o melhor centróide
      grupo = grupos[imaxs] if imaxs>=0 else -1
      sim = maxs*100 if imaxs>=0 else 0
      return grupo, int(sim)
  
    def agrupar(self, primeiro=True):
      grupos = self.dados['grupo']
      centroides = self.dados['centroide']
      passo = 'Criando centróides' if primeiro else 'Reorganizando similares'
      for i, (g,c) in enumerate(zip(grupos,centroides)):
          UtilDocs.progress_bar(i+1,len(grupos),f'{passo}')
          if g==-1 or c==0:
            v = self.dados.iloc[i]['vetor_np']
            # identifica o melhor centróide para o vetor
            g,s = self.melhor_grupo(v)
            if g >=0:
              self.dados.at[i,'grupo'] = g
              self.dados.at[i,'similaridade'] = s
            else:
              # não tendo um melhor centróide, cria um novo grupo
              g = max(self.dados['grupo']) +1
              self.dados.at[i,'grupo'] = g
              self.dados.at[i,'similaridade'] = 100
              self.dados.at[i,'centroide'] = 1
      if primeiro:
         # um segundo passo é feito para corrigir o centróide de quem ficou ente um grupo e outro
         # buscando o melhor dos centróides dos grupos que poderia pertencer
         self.agrupar(False)
         # corrige os grupos órfãos e renumera os grupos
         self.dados['grupo'] = [f'tmp{_}' for _ in self.dados['grupo']]
         grupos = Counter(self.dados['grupo'])
         #print('Grupos e quantidades: ', list(grupos.items()))
         ngrupo = 1
         for grupo,qtd in grupos.items():
             if qtd==1:
                self.dados.loc[self.dados['grupo'] == grupo, 'similaridade'] = 0
                self.dados.loc[self.dados['grupo'] == grupo, 'centroide'] = 0
                self.dados.loc[self.dados['grupo'] == grupo, 'grupo'] = -1
             else:
                self.dados.loc[self.dados['grupo'] == grupo, 'grupo'] = ngrupo
                ngrupo +=1
         # ordena pelos grupos
         self.dados['tmp_ordem_grupos'] = [g if g>=0 else float('inf') for g in self.dados['grupo']]
         self.dados.sort_values(['tmp_ordem_grupos','similaridade','centroide'], ascending=[True,False, False], inplace=True)
         self.dados.drop('tmp_ordem_grupos', axis='columns', inplace=True)
      if 'hash' in self.dados.columns:
        # se a vetorização foi feita com o método interno do AgrupamentoRapido, ele já gera o hash do texto
        # varre os grupos para indicar se o texto é idêntico ao centróide
        # os dados vão chegar aqui ordenados pelo centróide, 
        # então o primeiro de cada grupo é o hash de comparação
        _hash_centroide = 0
        _identicos = []
        for _,row in self.dados.iterrows():
            if row['centroide'] == 1:
                _hash_centroide = row['hash']
                _identicos.append('Sim')
                continue
            if row['grupo'] <= 0:
                _identicos.append('')
                continue
            if _hash_centroide == row['hash']:
                _identicos.append('Sim')
            else:
                _identicos.append('Não')
        self.dados['idêntico'] = _identicos

    def carregar_arquivos_para_dados(self, pasta_arquivos):
        # devolve um dicionário com os dados dos arquivos
        if (not pasta_arquivos) or (not os.path.isdir(pasta_arquivos)):
           return []
        lista = UtilDocs.listar_arquivos(pasta_arquivos)
        lista.sort()
        def _carregar(i):
            arquivo = lista[i]
            texto = UtilDocs.carregar_arquivo(arquivo)
            arquivo = os.path.split(arquivo)
            lista[i] = {'pasta': arquivo[0], 'nome': arquivo[1], 'texto': texto}
        UtilDocs.map_thread(_carregar, lista = range(len(lista)), n_threads=10)
        return lista
         
    # vetoriza os dados que não possuem a chave 'vetor' ou o vetor é None     
    # vetoriza e cria o hash do texto para identificar idêntivos
    def vetorizar_textos_dados(self, dados, pasta_modelo):
        if dados is None or len(dados) == 0:
           return
        # se for para vetorizar, carrega o modelo doc2vecrapido ou doc2bertrapido
        modelo = modelo_generico(pasta_modelo) 

        # não existindo arquivo de cache de vetores, faz todo o looping de vetorização
        if not self.arquivo_cache_vetores:
           modelo.vetorizar_dados(dados, epocas = self.epocas_vetorizacao)
           return 
         
        # se existir arquivo de cache de vetores
        # faz um looping de BATCH_VETORIZACAO documentos para ir gravando no cache
        # e conseguir aproveitar o que foi feito caso ocorra um erro de memória, por exemplo
        self.__carregar_cache__()
        self.__carregar_cache_ids__(dados) # vetores para os dados se existirem
        pos = 0
        while pos < len(dados):
              fim = min(pos + 1000, len(dados))
              self.printlog(f'Analisando cache [{pos}:{fim}]/{len(dados)} registros')
              modelo.vetorizar_dados(dados[pos:fim], epocas = self.epocas_vetorizacao)
              self.__gravar_cache_ids__(dados[pos:fim]) # gravar cache de vetores se existir
              pos = fim
        self.printlog(f'Analise de cache finalizada para {len(dados)}: {self.arquivo_cache_vetores}')
        # limpa o cache da memória
        self.cache_vetores = {}

    # arquivo sem extensão, inclui a similaridade e a extensão 
    def plotar(self, show_plot=True, arquivo_saida = None, similaridade = None):
      if len(self.dados) ==0:
         return
      # ajusta os x,y
      if not 'x' in self.dados.columns:
         # verifica se tem 2 dimensões
         if len(self.dados['vetor'][0]) >2:
            self.printlog(f'Reduzindo dimensões para plotagem de {len(self.dados["vetor"][0])}d para 2d')
            tsne_model = TSNE(n_components=2, init='pca', method='exact', n_iter=1000)
            vetores_2d = tsne_model.fit_transform(list(self.dados['vetor_np']) )
            x,y = zip(*vetores_2d)
         else:
            x,y = zip(*self.dados['vetor_np'])
         self.dados['x'] = x
         self.dados['y'] = y
      if arquivo_saida:
         plt.figure(dpi=300, figsize=(15,15))
      else:
        plt.figure(figsize=(13,13))
      sns.set_theme(style="white")
      grupos = list(set(self.dados['grupo']))
      custom_palette = sns.color_palette("Set3", len(grupos))
      custom_palette ={c:v if c >=0 else 'k' for c,v in zip(grupos,custom_palette)}
      #centroides
      tamanhos = [100 if t==1 else 50 if s==0 else 20 for t,s in zip(self.dados['centroide'],self.dados['similaridade']) ]
      sns.scatterplot( x="x", y="y", data=self.dados, hue='grupo', legend=False,  s = tamanhos, palette=custom_palette)
      if arquivo_saida:
         _arquivo_saida = self.get_nome_arquivo_saida(arquivo_saida=arquivo_saida, similaridade=similaridade, extensao = '.png')
         plt.savefig(f'{_arquivo_saida}')
         self.printlog(f'Plot finalizado e gravado em "{_arquivo_saida}"')
      if not show_plot:
         plt.close()
      return plt

    # arquivo sem extensão, inclui a similaridade e a extensão 
    def to_excel(self, arquivo_saida, similaridade = None):
         _arquivo_saida = self.get_nome_arquivo_saida(arquivo_saida=arquivo_saida, similaridade=similaridade)
         self.printlog(' construindo planilha de dados ... ')
         colunas_iniciais = ['centroide','grupo','similaridade','idêntico']
         colunas_iniciais = [_ for _ in colunas_iniciais if _ in self.dados.columns]
         colunas_fora = ['texto','vetor','vetor_np','hash'] + colunas_iniciais
         if self.coluna_texto and 'texto' in self.dados:
            self.dados['trechos'] = [_[:150] for _ in self.dados['texto'] ]
         colunas = []
         for coluna in self.dados.columns:
               _coluna = coluna.lower()
               if _coluna in colunas_fora:
                  continue
               elif _coluna.startswith('id'):
                  pos = 1
               elif _coluna in ['descricao','descrição','rótulo','rotulo']:
                  pos = 2
               elif _coluna in ['trecho']:
                  pos = 3
               elif _coluna.startswith('desc'):
                  pos = 4
               else:
                  pos = 5
               colunas.append((pos, coluna))
         colunas.sort()
         colunas = [_[1] for _ in colunas]
         colunas = colunas_iniciais + colunas
         self.printlog(f' finalizando arquivo excel para {_arquivo_saida}')
         #self.dados.to_excel(_arquivo_saida,
         #                     sheet_name=f'Agrupamento de arquivos',
         #                     index = False, columns=colunas)
         tp = UtilPandasExcel(_arquivo_saida)
         _nome_plan = 'Agrupamento Rápido'
         _dados = self.dados[colunas]
         tp.write_df(_dados, sheet_name=_nome_plan,auto_width_colums_list=True, columns_titles=colunas)
         # identificando a linha seguinte ao fim dos grupos
         _grupos = list(_dados['grupo'])
         _linha_orfao = _grupos.index(-1)
         _grupos = list(_dados['grupo'])
         _linha_orfao = _grupos.index(-1)
         _linha_orfao = len(_dados) if _linha_orfao <0 else _linha_orfao
         # colorindo o número dos grupos por par e ímpar
         str_cells = tp.range_cols(first_col=0, last_col=1, first_row=1, last_row=_linha_orfao +1)
         tp.conditional_value_color(sheet_name = _nome_plan, cells=str_cells, valor =  '=MOD($B1, 2)=0', cor = tp.COR_AZUL_CLARO)
         str_cells = tp.range_cols(first_col=3, last_col=len(colunas)-1,first_row=1, last_row=_linha_orfao +1)
         tp.conditional_value_color(sheet_name = _nome_plan, cells=str_cells, valor =  '=MOD($B1, 2)=0', cor = tp.COR_AZUL_CLARO)
         # sem grupo
         str_cells = tp.range_cols(first_col=0, last_col=len(colunas)-1,first_row=_linha_orfao+2, last_row=len(_dados)+1)
         tp.conditional_value_color(sheet_name = _nome_plan, cells=str_cells, valor =  '1=1', cor = tp.COR_CINZA)
         # idêntico
         str_cells = tp.range_cols(first_col=3, last_col=3,first_row=1, last_row=_linha_orfao +1)
         tp.conditional_value_color(sheet_name = _nome_plan, cells=str_cells, valor =  'D1="Sim"', cor = tp.FONTE_VERDE)
         # colorindo as similaridades
         str_cells = tp.range_cols(first_col=2, last_col=2,first_row=1, last_row=_linha_orfao +1)
         tp.conditional_color(sheet_name = _nome_plan, cells=str_cells)
         #tp.colorir_coluna(sheet_name=_nome_plan, idx_coluna = 1, funcao = lambda x: x>=0 and x%2==0)

         #tp.header_formatting = False
         tp.save()         
         self.printlog(f'Agrupamento finalizado e gravado em "{_arquivo_saida}"')
         return _arquivo_saida

    # cria um nome de arquivo de saída com base na similaridade, pasta opcional e complemento opcional
    # exemplo complemento: agrupamento {complemento} sim 80%.xslx
    # exemplo pasta: agrupamento {pasta_folha} sim 80%.xslx
    # exemplo arquivo: {arquivo} sim 80%.xslx
    @classmethod
    def get_nome_arquivo_saida(cls, similaridade = None, arquivo_saida = None, pasta = None, complemento = None, extensao = '.xlsx'):
        # o arquivo é válido, retorna ele mesmo
        if str(arquivo_saida).lower().endswith(extensao):
           return arquivo_saida
        # constrói um arquivo com o nome base enviado 
        if similaridade is None:
           _sim = ''
        elif 0 <= similaridade < 1:
           _sim = f' sim {round(similaridade * 100)}' 
        elif 1 <= similaridade <= 100:
           _sim = f' sim {similaridade}'
        else:
           _sim = f' sim {similaridade}'

        if str(extensao) and str(extensao)[0] != '.':
           extensao = f'.{extensao}'
        if complemento:
           return f'./agrupamento {complemento}{_sim}{extensao}'
        if pasta:
            comp = os.path.split(pasta)[-1]
            return f'./agrupamento {comp}{_sim}{extensao}'
        if arquivo_saida:
           _arquivo = os.path.splitext(arquivo_saida)[0]
           return f'./{_arquivo}{_sim}{extensao}'
        return f'./agrupamento{_sim}{extensao}'

    # cria um dataframe com os grupos, exporta para o excel (arquivo_saida) e retorna o dataframe
    # textos = True/False - inclui uma parte do texto do documento no dataframe
    @classmethod
    def agrupar_arquivos(self, pasta_modelo, pasta_arquivos, arquivo_saida = '', 
                         similaridade = 90, epocas = 100, plotar=False, 
                         arquivo_cache = None,
                         coluna_texto = False):
        assert os.path.isdir(pasta_arquivos), 'A pasta de arquivos não e válida'

        util = AgrupamentoRapido(pasta_arquivos=pasta_arquivos, 
                                 pasta_modelo=pasta_modelo, 
                                 similaridade=similaridade,
                                 epocas_vetorizacao=epocas,
                                 arquivo_cache_vetores=arquivo_cache,
                                 arquivo_saida_excel=arquivo_saida,
                                 plotar = bool(plotar),
                                 coluna_texto = coluna_texto)
        util.printlog('Agrupamento finalizado')
        return util
        
    def printlog(self, msg, destaque = False):
        UtilDocs.printlog('AgrupamentoRapido', msg, destaque=destaque)

    def __carregar_cache__(self):
        if not (self.arquivo_cache_vetores):
           self.cache_vetores = {}
           return 
        horas_cache = self.idade_arquivo_horas(self.arquivo_cache_vetores) 
        # cache com mais de 30 dias serão descartados
        if (horas_cache < 0) or (horas_cache > 24 * 30):
           self.cache_vetores = {}
           self.printlog('Arquivo de cache antigo ou inexistente, iniciando um novo cache')
           return
           
        with open(self.arquivo_cache_vetores, 'r') as f:
              cache = f.read()
              try:
                 cache = json.loads(cache) 
              except:
                 self.printlog(f'Arquivo de cache inválido {self.arquivo_cache_vetores} ... cache reiniciado', destaque=True)
                 cache = {}
        self.cache_vetores = cache
        self.printlog(f'Arquivo de cache carregado com {len(self.cache_vetores)} vetores: {self.arquivo_cache_vetores}') 

    def __carregar_cache_ids__(self, dados):
        ''' para o cache funcionar, tem que existir a chave hash nos dados
            pois ela liga o vetor à linha de dados 
        '''
        if not any(self.cache_vetores):
           return 0
        qtd_cache = 0 
        def _gerar_hashs(i):
            if (not dados[i].get('hash')) and 'texto' in dados[i]:
               dados[i]['hash'] = UtilDocs.hash(dados[i]['texto'])
        self.printlog(f' - gerando hash dos {len(dados)} textos para uso do cache')
        UtilDocs.map_thread(func = _gerar_hashs, lista = range(len(dados)), n_threads=10)
        # carrega o vetor do cache
        qtd_sem_hash, qtd_sem_vetor = 0, 0
        for linha in dados:
            if not linha.get('hash'):
               qtd_sem_hash += 1
               continue 
            vetor = self.cache_vetores.get(linha['hash'])
            if vetor is None:
               qtd_sem_vetor += 1
               continue
            linha['vetor'] = vetor
            qtd_cache += 1
        if qtd_cache + qtd_sem_vetor + qtd_sem_hash > 0 :
           UtilDocs.print_linha()
           if qtd_cache > 0:
              self.printlog(f'Número de vetores carregados do cache: {qtd_cache}')
           if qtd_sem_hash + qtd_sem_vetor > 0:
              self.printlog(f'Número de dados sem hash = {qtd_sem_hash} e número de vetores vazios = {qtd_sem_vetor}')
           UtilDocs.print_linha()
        return qtd_cache > 0 

    def __gravar_cache_ids__(self, dados):
        if not self.arquivo_cache_vetores:
           return
        # verifica se tem vetor nas linhas de dados e o hash do texto para associar ao vetor
        self.printlog('Gravando cache de vetores ...') 
        qtd = 0
        for linha in dados:
            if 'vetor' in linha and 'hash' in linha \
               and linha['hash'] and linha['vetor'] is not None:
               self.cache_vetores[linha['hash']] = linha['vetor']
               qtd += 1
        with open(self.arquivo_cache_vetores, 'w') as f:
             f.write(json.dumps(self.cache_vetores))
        self.printlog(f'Cache de {qtd} vetores atualizados e {len(self.cache_vetores)} gravados no arquivo {self.arquivo_cache_vetores}')         
        return True 

    @classmethod
    def data_arquivo(cls, arquivo):
        if not os.path.isfile(arquivo):
            return None
        return datetime.fromtimestamp(os.path.getmtime(arquivo))

    @classmethod
    def idade_arquivo_horas(cls, arquivo):
        data = cls.data_arquivo(arquivo)
        if not data:
           return -1
        return (datetime.now() - data).total_seconds() / 3600

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Pasta do modelo')
    parser.add_argument('-modelo', help='pasta contendo o modelo - padrao meu_modelo', required=False)
    parser.add_argument('-textos', help='pasta contendo os textos que serão agrupados - padrao ./textos', required=False)
    parser.add_argument('-sim', help='similaridade padrão 90%', required=False)
    parser.add_argument('-epocas', help='épocas para inferir o vetor padrão 100', required=False)
    parser.add_argument('-plotar', help='plota um gráfico com a visão 2d do agrupamento', required=False, action='store_const', const=1)
    parser.add_argument('-texto', help='inclui uma coluna "trechos" com parte do texto no resultado', required=False, action='store_const', const=1)
    parser.add_argument('-nocache', help='Não cria o arquivo de cache para os textos', required=False, action='store_const', const=1)
    parser.add_argument('-saida', help='nome do arquivo de saída - opcional', required=False)

    args = parser.parse_args()

    similaridade = int(args.sim or 90)
    epocas = int(args.epocas or 100)
    epocas = 1 if epocas<1 else epocas
    plotar = args.plotar
    coluna_texto = args.texto

    PASTA_MODELO = args.modelo or './meu_modelo'

    PASTA_TEXTOS = args.textos or './textos'
    if (not os.path.isdir(PASTA_TEXTOS)):
        print(f'ERRO: pasta de textos não encontrada em "{PASTA_TEXTOS}"')
        exit()

    if args.nocache:
       arquivo_cache = None 
    else:
       _modelo = os.path.basename(os.path.normpath(PASTA_MODELO))
       arquivo_cache = f'__cache_agrupamento_{_modelo}__.json'

    modelo_generico(PASTA_MODELO, testar = True)

    arquivo_saida = args.saida or 'agrupamento'

    util = AgrupamentoRapido.agrupar_arquivos(pasta_modelo=PASTA_MODELO, 
                                          pasta_arquivos=PASTA_TEXTOS, 
                                          similaridade=similaridade,
                                          epocas = epocas,
                                          plotar = plotar,
                                          arquivo_saida = arquivo_saida,
                                          arquivo_cache = arquivo_cache,
                                          coluna_texto = coluna_texto)


