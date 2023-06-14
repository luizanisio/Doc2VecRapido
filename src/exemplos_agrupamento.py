# -*- coding: utf-8 -*-
from util_agrupamento_rapido import AgrupamentoRapido
from util_doc2vec_rapido import Doc2VecRapido
from exemplos import carregar_paragrafos
import os
import pandas as pd
import json

#######################################################################
# Exemplo de uso da classes: Doc2VecRapido e AgrupamentoRapido
# Esse código, dicas de uso e outras informações: 
#   -> https://github.com/luizanisio/Doc2VecRapido
# Luiz Anísio 
#######################################################################

def agrupar_pasta():
    pasta = './textos_legislacoes'
    modelo = './meu_modelo'
    similaridade = 90
    plotar = False
    arquivo_saida = './agrupamento.xlsx'
    arquivo_cache = './tmpcache/cache.json'
    os.makedirs(os.path.split(arquivo_cache)[0], exist_ok=True)
    util = AgrupamentoRapido.agrupar_arquivos(pasta_modelo=modelo, 
                                          pasta_arquivos=pasta, 
                                          similaridade=similaridade,
                                          epocas = 100,
                                          plotar = plotar,
                                          arquivo_saida = arquivo_saida,
                                          arquivo_cache = arquivo_cache,
                                          coluna_texto = True)
    
def agrupar_arquivo_com_vetores():
    arquivo = './vetorizados/parag_legislacoes.json'
    df = pd.read_json(arquivo, lines = True)
    dados = df.to_dict(orient='records')
    modelo = ''
    similaridade = 90
    plotar = False
    arquivo_saida = './agrupamento.xlsx'
    arquivo_cache = './tmpcache/cache.json'
    print(f'Preparando para agrupar {len(dados)} documentos')    
    os.makedirs(os.path.split(arquivo_cache)[0], exist_ok=True)
    util = AgrupamentoRapido(dados = dados,
                             pasta_modelo= modelo, 
                             similaridade=similaridade,
                             plotar = plotar,
                             coluna_texto = True,
                             arquivo_saida_excel = arquivo_saida)

def criar_vetorizados():
    textos_tags = carregar_paragrafos('./textos_legislacoes', 100)
    print(f'Vetorizando {len(textos_tags)} parágrafos das legislações')
    #textos_tags = textos_tags[:100]
    docs = [{'texto': texto} for texto, _ in textos_tags]
    dv = Doc2VecRapido(pasta_modelo='./meu_modelo')
    dv.vetorizar_dados(docs)
    arq_saida = './vetorizados/parag_legislacoes.json'
    os.makedirs(os.path.split(arq_saida)[0], exist_ok=True)
    with open(arq_saida,'w') as f:
         for d in docs:
             f.write(json.dumps(d) + '\n')
    print(f'Arquivo com {len(docs)} documentos vetorizados criado com sucesso!')


if __name__ == '__main__':
    # exemplo 1 - agrupar arquivos de um apasta de textos
    # agrupar_pasta()

    # exemplos 2 - agrupar textos de um dataframe
    agrupar_arquivo_com_vetores()

    # exemplo 3 - criar um arquivo com textos vetorizados 
    # feito para gerar dados para o exemplo 2
    # criar_vetorizados()


#1951 segundos para 