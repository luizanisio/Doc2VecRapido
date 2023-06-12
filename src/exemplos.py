# -*- coding: utf-8 -*-

#######################################################################
# Exemplo de uso da classes: Doc2VecRapido
# Esse código, dicas de uso e outras informações: 
#   -> https://github.com/luizanisio/Doc2VecRapido
# Luiz Anísio 
#######################################################################

from util_doc2vec_rapido import Doc2VecRapido,DocumentosDoArquivo
from util_doc2util import UtilDocs
import pandas as pd
import os

# o primeiro exemplo, treinar o modelo com os arquivos de uma pasta
# os arquivos serão carregados em memória e o modelo treinado
def treinar_de_pasta():
   PASTA_MODELO = './meu_modelo'
   PASTA_TEXTOS = './textos_grupos'
   EPOCAS = 50
   dv = Doc2VecRapido(pasta_modelo=PASTA_MODELO, 
                      documentos=PASTA_TEXTOS, 
                      epochs=EPOCAS, 
                      skip_gram = True,
                      stemmer = False,
                      strip_numeric = True,
                      arq_tagged_docs = None,
                      workers = 30 )
   dv.treinar()
   print('Treino concluído')
   testar_modelo()

# para criar um arquivo de treino já pré-processado com 
# milhares ou milhões de documentos, pode-se carregar os textos
# de um ou mais dataframes 
# no exemplo depois desse pode ser criado com uma pasta de textos
def criar_arquivo_treino_de_dataframes():
    arq1 = './dataframes/arquivo.feather'
    arq2 = './dataframes/outro_arquivo.json'
    df1 = pd.read_feather(arq1)
    df2 = pd.read_json(arq2, lines=True)
    # nome do arquivo de treino que será gerado
    arq_saida = './dataframes/textos_para_treino.json'
    # apaga o arquivo de saída se existir
    if os.path.isfile(arq_saida):
        os.remove(arq_saida)
    # aqui o arquivo config já é criado para o modelo 
    # na pasta indicada com os parâmetros do tokenizador
    dv = Doc2VecRapido(pasta_modelo='./meu_modelo', 
                       strip_numeric=True, 
                       stemmer=False, 
                       skip_gram=True)        
    docs = DocumentosDoArquivo(arq_saida, dv)
    # aqui os dataframes são processados e inseridos no arquivo de treino    
    docs.incluir_textos([df1,df2], coluna_texto='texto', coluna_tags='rotulos')    
    print(f'Arquivo de treino criado: ', arq_saida)

# para criar um arquivo de treino já pré-processado com 
# milhares ou milhões de documentos de uma ou mais pastas de textos
# pode ser ajustado para pegar documentos de qualquer lugar
def criar_arquivo_treino_de_textos():
    textos, tags = UtilDocs.carregar_documentos('./textos_grupos')
    # aqui o arquivo config já é criado para o modelo 
    # na pasta indicada com os parâmetros do tokenizador
    dv = Doc2VecRapido(pasta_modelo='./meu_modelo', 
                       strip_numeric=True, 
                       stemmer=False, 
                       skip_gram=True)        
    # aqui os textos são processados e inseridos no arquivo de treino
    arq_saida = './dataframes/textos_para_treino_textos.json'
    # apaga o arquivo de saída se existir
    if os.path.isfile(arq_saida):
        os.remove(arq_saida)
    print('Criando arquivo de treino ... ')
    docs = DocumentosDoArquivo(arq_saida, dv)
    for (texto, _tags) in zip(textos, tags):
        docs.incluir_texto(texto, tags = _tags)
    print(f'Arquivo de treino criado: ', arq_saida)

# exemplo de como treinar o modelo usando um arquivo de treino preparado
# anteriormente com tokens e tags
def treinar_arquivo_treino():
   PASTA_MODELO = './meu_modelo'
   EPOCAS = 50
   arquivo_treino = './dataframes/textos_para_treino_textos.json'
   # entendendo que o arquivo config já existe com as informações
   # da tokenização
   dv = Doc2VecRapido(pasta_modelo=PASTA_MODELO, 
                      arq_tagged_docs= arquivo_treino,
                      bloco_tagged_docs= 1000,
                      epochs=EPOCAS,
                      workers = 30 )
   dv.treinar()
   print('Treino concluído')
   testar_modelo()

# apenas carrega e testa o modelos
def testar_modelo():       
   PASTA_MODELO = './meu_modelo'
   dv = Doc2VecRapido(pasta_modelo=PASTA_MODELO)

   texto_1 = 'esse é um texto de teste para comparação'
   texto_2 = 'esse é outro texto para comparação'

   vetor = dv.vetor(texto_1)
   print(f'Vetor do texto 1: {vetor[:2]} ... {vetor[-2:]}')        

   sim = dv.similaridade(texto_1, texto_2)
   print('Texto 1: ', texto_1)
   print('Texto 2: ', texto_2)
   print(f'Similaridade entre os textos: {sim}') 

# como criar um dataframe de documentos 
# não tem necessidade pois a classe pode carregar arquivos, é só
# um exemplo que criei para gerar os dados para os outros exemplos
def criar_dataframes():
    textos_tags = carregar_paragrafos('./textos_legislacoes', 100)
    df = pd.DataFrame([{'texto': texto, 'rotulos' : tags } for (texto, tags) in textos_tags])
    print('Textos: ', len(df))
    arq1 = './dataframes/arquivo.feather'
    arq2 = './dataframes/outro_arquivo.json'
    df = df.reset_index()
    df1 = df[:5000]
    df1.to_feather(arq1)
    df2 = df[5000:10000]
    df2.to_json(arq2, lines=True, orient='records')

# carrega os parágrados dos documentos de uma pasta que contenham
# um tamanho mínimo de caracteres
def carregar_paragrafos(pasta, tamanho_minimo = 50):
    textos, tags = UtilDocs.carregar_documentos(pasta)
    paragrafos = []
    for texto, _tags in zip(textos, tags):
        sentencas = texto.split('\n')
        for sentenca in sentencas:
            sentenca = sentenca.strip()
            if len(sentenca) >= tamanho_minimo:
                paragrafos.append((sentenca, _tags))
    return paragrafos


if __name__ == '__main__':
    # exemplo 1 - pata de textos
    treinar_de_pasta()

    # exemplo 2 preparar arquivo de treino 
    # carregando de dataframes
    criar_arquivo_treino_de_dataframes()

    # exemplo 3 preparar arquivo de treino 
    # carregando de uma pasta de textos
    criar_arquivo_treino_de_textos()

    # treinar usando um arquivo de treino
    treinar_arquivo_treino()
    
    # testar um modelo
    testar_modelo()
    
    # usado para gerar uma massa para esses testes
    #criar_dataframes()
