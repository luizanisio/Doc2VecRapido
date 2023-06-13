# -*- coding: utf-8 -*-

#######################################################################
# Classes: 
# Doc2VecRapido : permite carregar um modelo Doc2Vec treinado para aplicar em documentos de uma determinata área de conhecimento
#                 permite ajustar as configurações do treinamento usando o arquivo config.json
#                 permite treinar na versão 3.5.x e 4.0.x do gensim
# Esse código, dicas de uso e outras informações: 
#   -> https://github.com/luizanisio/Doc2VecRapido
# Luiz Anísio 
# 14/01/2023 - disponibilizado no GitHub  
#######################################################################
# Dicas de migração gensim 3.5 para 4.0: 
#   https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4

import logging

import os
import json

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from numpy import linalg
from scipy.spatial import distance

from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_numeric 
from nltk.stem.snowball import SnowballStemmer
from copy import deepcopy

from util_doc2util import UtilDocs
import pandas as pd
from typing import Union
from time import time
from unidecode import unidecode
import random

'''
  config.json = {"vector_size": 300, "strip_numeric":true, "stemmer":false, "min_count": 5 , "token_br": true}

  strip_numeric = remove números
  stemmer = utiliza o stemmer dos tokens
  min_count = ocorrência mínima do token no corpus para ser treinado
  token_br = cria o token #br para treinar simulando a quebra de parágrafos
  vector_size = número de dimensões do vetor que será treinado
'''

#####################################
class Config():
    def __init__(self, arquivo = None, strip_numeric = False, stemmer = False,
                 vector_size = 300, window = 5, min_count = 5, skip_gram = None, 
                 skip_gram_window = 10, token_br = True)  -> None:
        self.strip_numeric = strip_numeric
        self.stemmer = stemmer
        self.vector_size = vector_size
        self.min_count = min_count
        self.window = window
        self.skip_gram = False # padrão
        self.skip_gram_window = skip_gram_window
        self.max_total_epocas = 0
        self.token_br = token_br
        self.log_treino_epocas = 0
        self.log_treino_vocab = 0
        if arquivo:
           self.carregar(arquivo) 
        if skip_gram != None:
           self.skip_gram = skip_gram # sobrepõe o config se for preenchido 
    
    def __str__(self) -> str:
        res = [f'{c}={v}' for c,v in self.as_dict().items()]
        return ', '.join(res)

    def as_dict(self) -> dict:
        return self.__dict__

    def carregar(self, arquivo):
        if not os.path.isfile(arquivo):
           return 
        with open(arquivo,'r') as f:
             _config = f.read()
             if _config.find('{') >= 0 and _config.find('}')>1:
                try:
                   _config = json.loads(_config)
                   self.vector_size = _config.get('vector_size',self.vector_size)
                   self.stemmer = _config.get('stemmer',self.stemmer)
                   self.strip_numeric = _config.get('strip_numeric',self.strip_numeric)
                   self.min_count = _config.get('min_count',self.min_count)
                   self.token_br = _config.get('token_br',self.token_br)
                   self.window = _config.get('window',self.window)
                   self.skip_gram = _config.get('skip_gram',self.skip_gram)
                   self.skip_gram_window = _config.get('skip_gram_window',self.skip_gram_window)
                   self.max_total_epocas = _config.get('max_total_epocas',self.max_total_epocas)
                   self.log_treino_epocas = _config.get('log_treino_epocas',0)
                   self.log_treino_vocab = _config.get('log_treino_vocab',0)
                   self.print(f'> Config: carregado: {self.config}')
                except:
                   return

    def gravar_config(self, arquivo, parametros_modelo = None):
        _cfg = deepcopy(self.as_dict())
        if parametros_modelo != None and type(parametros_modelo) is dict and any(parametros_modelo):
           _cfg = deepcopy(self.as_dict())
           _cfg.update({'model':parametros_modelo})
        with open(arquivo, 'w') as f:
            f.write(json.dumps(_cfg, indent=2))

class Doc2VecRapido():
    CORT_SIM_VOCAB_EXEMPLOS = 0.7 # corte para gereação do arquivo de exemplos de similaridade
    PASTA_PADRAO = 'd2vmodel'
    ARQUIVO_MODELO = 'doc2vecrapido.d2v'

    def __init__(self, pasta_modelo = PASTA_PADRAO, 
                       documentos = [], tags = [], 
                       vector_size = 300, 
                       min_count = 5, 
                       epochs = 1000,
                       arq_tagged_docs: str = None,
                       bloco_tagged_docs: int  =10000,
                       strip_numeric = True,
                       stemmer = False,
                       skip_gram = False,
                       workers = -1) -> None:
        # configura arquivos da classe
        #if pasta_modelo and pasta_modelo[-1] in ['\\','/']:
        #   pasta_modelo = pasta_modelo[:-1]            
        self.workers = workers if workers and workers > 0 else 10
        self.pasta_modelo = os.path.realpath(pasta_modelo)
        self.nome_modelo = os.path.splitext(os.path.basename(self.pasta_modelo))[-1]
        self.arquivo_config = os.path.join(self.pasta_modelo,'config.json')
        self.arquivo_modelo = os.path.join(pasta_modelo, self.ARQUIVO_MODELO)
        self.arquivo_vocab = os.path.join(pasta_modelo,'vocab_treinado.txt')
        self.arquivo_vocab_sim = os.path.join(pasta_modelo,'vocab_similares.txt')
        self.arquivo_tags = os.path.join(pasta_modelo, 'tags_treinadas.txt')
        self.arquivo_stopwords = os.path.join(pasta_modelo,'stopwords.txt')
        self.arq_tagged_docs = None
        self.__len__docs__ = -1
        # prepara para iterar nos documentos já tokenizados e com as tags já tratadas
        if arq_tagged_docs is not None and os.path.isfile(arq_tagged_docs):
           self.arq_tagged_docs = arq_tagged_docs
        self.bloco_tagged_docs = bloco_tagged_docs
        # carrega as configurações se existirem
        self.config = Config(arquivo=self.arquivo_config,
                             min_count = min_count, 
                             vector_size=vector_size, 
                             strip_numeric=strip_numeric, 
                             stemmer=stemmer,
                             skip_gram=skip_gram)
        self.model = None
        self.documentos_carregados = 0
        self.tagged_docs = [] 
        self.epochs = None
        self.stop_words_usuario = self.STOP_WORDS_BR
        # arquivo com stopwords
        if os.path.isfile(self.arquivo_stopwords):
           self.stop_words_usuario = set(preprocess_string(self.carregar_arquivo(self.arquivo_stopwords), self.CUSTOM_FILTERS_NUM)) 
        
        # sem documentos, tenta carregar o modelo
        if os.path.isfile(self.arquivo_modelo):
            self.printlog(f'Carregando modelo {self.pasta_modelo}')
            self.model = Doc2Vec.load(self.arquivo_modelo)
            _dados_modelo = self.dados_modelo_por_versao(self.model)
            # atualiza o config se precisar pois alguns parâmetros não mudam depois do treinamento
            # corrige o config para ficar consistente com o modelo
            if (self.config.vector_size != self.model.wv.vector_size) or \
               (self.config.log_treino_vocab != _dados_modelo.get('vocab_length',0) ) or \
               (self.config.window != self.model.window) or \
               (self.config.skip_gram != self.model.sg == 1):  
                self.config.vector_size = self.model.wv.vector_size
                self.config.window = self.model.window
                self.config.skip_gram = self.model.sg == 1
                self.config.skip_gram_window = self.model.window if self.model.sg == 1 else self.config.skip_gram_window
                self.config.log_treino_vocab = _dados_modelo.get('vocab_length',0)
                self.config.gravar_config(self.arquivo_config, self.get_parametros_modelo())
            self.printlog(f'Modelo carregado com {self.config.log_treino_vocab} termos e {self.config.vector_size} dimensões  \n - CONFIG: {self.config}', destaque=True)
        if (not os.path.isfile(self.arquivo_modelo)) and (not any(documentos)) and self.arq_tagged_docs is None:
           self.printlog(f'não foi possível encontrar o modelo "{self.arquivo_modelo}" ou documentos para treiná-lo',True)
        # verificar e criar a pasta para o modelo
        os.makedirs(self.pasta_modelo, exist_ok=True)
        self.min_count = self.config.min_count
        self.epochs = epochs
        if self.arq_tagged_docs is None:
            # caso documentos seja uma string, verifica se é uma pasta com documentos
            if type(documentos) is str:
               if os.path.isdir(documentos):
                  _docs, _tags = self.carregar_documentos(documentos)
                  self.tagged_docs = self.pre_processar(documentos =_docs, tags = _tags, retornar_tagged_doc = True)
               else:
                  msg = f'ERRO: Não foi possível encontrar a pasta de documentos: {documentos}'
                  raise Exception(msg)
            else:
                  self.tagged_docs = self.pre_processar(documentos =documentos, tags = tags, retornar_tagged_doc = True)
            self.documentos_carregados = len(self.tagged_docs)
            if self.documentos_carregados > 0:
               self.printlog(f'Pronto para treinar o modelo {self.pasta_modelo} com {self.documentos_carregados} documentos')
            else:
               if self.model is None:
                  self.printlog(f'Aguardando inclusão de documentos para treinar o modelo {self.pasta_modelo}') 
        else:
            self.printlog(f'Pronto para treinar o modelo:\n - {self.pasta_modelo} \n - com os TaggedDocs do arquivo {self.arq_tagged_docs}') 
        self.config.gravar_config(self.arquivo_config, self.get_parametros_modelo())

    @classmethod
    def apagar_modelo(cls, pasta_modelo):
        cls.printlog('Removendo arquivos do modelo...')
        arqs = [os.path.join(pasta_modelo, cls.ARQUIVO_MODELO),
                os.path.join(pasta_modelo, f'{cls.ARQUIVO_MODELO}.docvecs.vectors_docs.npy'),
                os.path.join(pasta_modelo, f'{cls.ARQUIVO_MODELO}.trainables.syn1neg.npy'),
                os.path.join(pasta_modelo, f'{cls.ARQUIVO_MODELO}.wv.vectors.npy')
                ]
        q = 0 
        for arq in arqs:
            if os.path.isfile(arq):
               os.remove(arq) 
               q+=1
        if q == 1:
           cls.printlog(f' - {q} documento excluído')  
        elif q > 1:
           cls.printlog(f' - {q} documentos excluídos')  
        else:
           cls.printlog(f' - nenhum documento excluído')  

    def __len__(self):
       if self.__len__docs__ >= 0:
          return self.__len__docs__
       if self.arq_tagged_docs:
          td = DocumentosDoArquivo(self.arq_tagged_docs)
          self.__len__docs__ = len(td)
       else:
          self.__len__docs__ = len(self.tagged_docs)  
       return self.__len__docs__  

    @classmethod  
    def processar_tags(self, tags, i = None):
        '''recebe uma string ou uma lista e ajusta os items para lista de strings, incluindo str(i) ou qtd documentos atual'''
        res = []
        if UtilDocs.is_iterable(tags):
           res = [str(_) for _ in tags] 
        elif tags is None or tags == '':
           pass
        elif type(tags) is str and tags and tags [0] == '[' and tags[-1] == ']':
           dados = json.loads(tags)
           res = [str(_) for _ in dados] 
        elif type(tags) in (str, int, float):        
           res = [str(tags)] 
        elif type(tags) is str and tags.find('\n') >=0:
           res = tags.split('\n') 
        elif type(tags) is str and tags.find(';') >=0:
           res = tags.split(';') 
        elif type(tags) is str and tags.find(',') >=0:
           res = tags.split(',') 
        else: 
           res = [str(tags)]
        _tag_i = [] if i is None else [str(i)]
        return res + _tag_i

    @classmethod
    def novo_config(cls, pasta_modelo):
        if not os.path.isdir(pasta_modelo):
            os.makedirs(pasta_modelo)
        arquivo = os.path.join(pasta_modelo, 'config.json')
        cf = Config(arquivo=arquivo)
        cf.gravar_config(arquivo=arquivo)
        print(f'Arquivo de configuração criado em {arquivo}')

    @classmethod
    def modelo_existe(cls, pasta_modelo):
        _pasta_modelo = os.path.basename(pasta_modelo)
        _nome_modelo = os.path.join(_pasta_modelo,'doc2vecrapido.d2v')
        return os.path.isfile(_nome_modelo)

    @classmethod
    def config_existe(cls, pasta_modelo):
        '''pode ser usado para criar um config ao chamar a primeira vez'''
        _pasta_modelo = os.path.basename(pasta_modelo)
        _nome_modelo = os.path.join(_pasta_modelo,'config.json')
        return os.path.isfile(_nome_modelo)

    @classmethod
    def printlog(self,msg, retornar_msg_erro=False, destaque = False):
        UtilDocs.printlog('Doc2VecRapido', msg, retornar_msg_erro, destaque)

    def pre_processar(self, documentos = [], tags=None, retornar_tagged_doc = True):
        # documentos como texto único, infere que tags é apenas dele ou não tem
        if type(documentos) is str:
           _tags = [tags] if type(tags) is list and any(tags) else []
           tokens =  self.pre_processar(documentos=[documentos], tags=_tags, retornar_tagged_doc=retornar_tagged_doc) 
           return tokens[0]
        # recebeu uma lista de documentos e tags
        # cada item de tags deve ser uma lista de strings/rotulos
        usar_tags = type(tags) is list and any(tags) and len(tags) >= len(documentos)
        res = []
        doc_tokens = self.preprocess_br(documentos)
        for i, tokens in enumerate(doc_tokens):
            #tokens = self.preprocess_br(line)
            if not retornar_tagged_doc:
                res.append(tokens)
            else:
                # Retorna o formato de treino
                if usar_tags:
                   _lista_tags_str = self.processar_tags(tags[i], i)
                   #print(f'lista de tags: {_lista_tags_str}')
                   res.append( TaggedDocument(tokens, _lista_tags_str) )
                else:
                   res.append( TaggedDocument(tokens, [str(i)]) )
        return res

    def dados_modelo_por_versao(self, modelo):
        if 'vocab' in modelo.wv.__dict__:
           # versão 3.5.0
           dados = {'ver': '3.5.x'}            
           dados['vocab'] = sorted([str(_) for _ in modelo.wv.vocab])
           dados['vocab_length'] = len(dados['vocab'])
        else:
           # versão 4.0.1  
           dados = {'ver': '4.x.x'}            
           dados['vocab'] = sorted([str(_) for _ in modelo.wv.key_to_index])
           dados['vocab_length'] = len(dados['vocab'])
        return dados

    def treinar(self):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        tagged_docs_iter = None
        if self.arq_tagged_docs and os.path.isfile(self.arq_tagged_docs):
           tagged_docs_iter = DocumentosDoArquivo(self.arq_tagged_docs, qtd_linhas_bloco=self.bloco_tagged_docs)
           self.printlog(f'Treinando modelo com TaggedDocs do arquivo {self.arq_tagged_docs} épocas \n - CONFIG: {self.config}', destaque=True)
        else:
           self.printlog(f'Treinando modelo com {self.documentos_carregados} documentos e {self.epochs} épocas \n - CONFIG: {self.config}', destaque=True)
        if self.model is None:
           self.config.skip_gram_window = max(1, self.config.skip_gram_window)
           self.config.window = max(1, self.config.window)
           self.config.vector_size = max(10, self.config.vector_size)
           self.config.min_count = max(1, self.config.min_count)
           model = Doc2Vec(vector_size=self.config.vector_size, 
                           min_count=self.min_count, 
                           epochs=self.epochs,
                           window = self.config.skip_gram_window if self.config.skip_gram else self.config.window,
                           workers=self.workers)
           if self.config.skip_gram:
              model.sg = 1 
           
           if tagged_docs_iter != None:
              self.printlog(f'Criando vocab para o novo modelo: {self.arq_tagged_docs}')
              model.build_vocab(tagged_docs_iter) 
           else: 
              self.printlog(f'Criando vocab para o novo modelo: documentos em memória')
              assert len(self.tagged_docs) > 0, 'Não foram encontrados documentos para treinamento'
              model.build_vocab(self.tagged_docs)
           if self.config.skip_gram:
              self.printlog(f'Treinando novo modelo: vector_size={self.config.vector_size} Skip-gram w={model.window}')
           else:
              self.printlog(f'Treinando novo modelo vector_size={self.config.vector_size} CBow w={model.window}')
           self.config.log_treino_epocas = 0 #reinicia a contagem de épocas
        else:
           model = self.model
           self.printlog(f'Atualizando modelo existente')
        # o treinamento é feito em blocos de 50 épocas para gravar o modelo a cada bloco
        total_epocas = self.epochs
        epocas_treinadas = self.config.log_treino_epocas
        # verifica o limite total de épocas
        if self.config.max_total_epocas > 0:
           if total_epocas + epocas_treinadas > self.config.max_total_epocas:
              total_epocas = self.config.max_total_epocas - epocas_treinadas 
        _dados_modelo = dict()
        while total_epocas > 0:
              bloco_epocas = 50 if total_epocas > 50 else total_epocas
              total_epocas -= bloco_epocas
              # no caso de um arquivo com tagged_docs, o treinamento será feito com o arquivo e com os documentos 
              # em tagged_docs se existirem
              if tagged_docs_iter != None:
                 self.printlog(f'Iniciando treinamento com documentos do arquivo: "{self.arq_tagged_docs}"')
                 model.train(tagged_docs_iter, 
                             total_examples=model.corpus_count, 
                             epochs=bloco_epocas)
              else: 
                 model.train(self.tagged_docs, 
                             total_examples=model.corpus_count, 
                             epochs=bloco_epocas)
              epocas_treinadas += bloco_epocas
              self.printlog(f'Gravando modelo após bloco com {bloco_epocas} épocas\n - total {epocas_treinadas} épocas treinadas \n - épocas restantes: {total_epocas}', destaque=True)
              self.config.log_treino_epocas = epocas_treinadas
              if not any(_dados_modelo):
                 _dados_modelo = self.dados_modelo_por_versao(model)
              self.config.log_treino_vocab = _dados_modelo['vocab_length']
              model.save(self.arquivo_modelo)
              self.config.gravar_config(self.arquivo_config, self.get_parametros_modelo(model)) # atualiza o config
        if not any(_dados_modelo):
            _dados_modelo = self.dados_modelo_por_versao(model)
        self.model = model 
        self.printlog(f'Gravando vocab treinado')
        # vocab treinado
        with open(self.arquivo_vocab,'w') as f:
             f.write('\n'.join(_dados_modelo.get('vocab',[])))
        # similares do vocab
        self.printlog(f'Gravando log de similaridade entre os termos')
        with open(self.arquivo_vocab_sim,'w') as f:
             for termo in _dados_modelo.get('vocab',[]):
                 ms = self.model.wv.most_similar(termo, topn=3)
                 ms = [f'{_[0]}={_[1]:.2f}' for _ in ms if _[1]>=self.CORT_SIM_VOCAB_EXEMPLOS]
                 if len(ms) > 0:
                    ms = ', '.join(ms)
                    f.write(f'{termo}: {ms}\n')
        self.printlog(f'Gravando lista de tags treinadas nos documentos')
        with open(self.arquivo_tags, 'w') as f:
             _tags_totais = set()  
             if tagged_docs_iter != None:  
               tagged_docs_iter.reiniciar()
               for td in tagged_docs_iter:
                 _tags_totais.update(td.tags)
               print(f' - tags totais do arquivo de treino: {len(_tags_totais)}')
             else:
               for td in self.tagged_docs:
                  _tags_totais.update(td.tags)
               print(f' - tags totais dos documentos em memória: {len(_tags_totais)}')
             _tags_totais = list(_tags_totais)
             # coloca as tags numéricas depois das strings
             _tags_totais.sort(key = lambda x:'zzz'+str(x).rjust(20) if str(x).isnumeric() or f'{x} '[0]=='#' else str(x).strip().lower())
             f.write('\n'.join([str(_) for _ in _tags_totais]))

        self.printlog(f'Modelo treinado com {self.config.log_treino_vocab} termos e {self.config.vector_size} dimensões')
        self.printlog(f'Modelo treinado com {self.epochs} épocas, totalizando {self.config.log_treino_epocas} desde sua criação.')
        if self.config.log_treino_epocas >= self.config.max_total_epocas:
           self.printlog(f'Treinamento atingiu o máximo de épocas configuradas: "{self.config.max_total_epocas}"')
        self.printlog(f'Modelo gravado em "{self.arquivo_modelo}"', destaque=True)

        # TODO: model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)


    def vetor(self, texto, epochs = 100, normalizar = True, retornar_tokens = False):
        assert self.model, 'Não foi treinado ou carregado um modelo para inferência do vetor!'
        tokens = self.pre_processar(documentos = texto, retornar_tagged_doc=False)
        if len(tokens) == 0:
            # sem tokens para vetorizar, joga o vetor no rodapé do espaço vetorial :)
            vetor = [1 for _ in range(self.model.wv.vector_size) ]
        else:
            vetor = self.model.infer_vector(tokens, epochs = epochs) 
        if normalizar:
           vetor = [float(f) for f in vetor / linalg.norm(vetor)]     
        else:
           vetor = [float(f) for f in vetor]
        if retornar_tokens:
           return vetor, tokens
        return vetor 

    def similaridade(self, texto1, texto2, epochs = 100):
        vetor1 = self.vetor(texto1, epochs=epochs, normalizar=False) 
        vetor2 = self.vetor(texto2, epochs=epochs, normalizar=False) 
        return 1- distance.cosine(vetor1,vetor2)

    # pré-processamento com números e stem sendo opcionais
    # o modelo carregado deve usar o mesmo processador do treino
    STEMMER = SnowballStemmer('portuguese')
    STOP_WORDS_BR = {'de', 'da', 'do', 'para', 'ao', 'a', 'e', 'i', 'o', 'u', 
                     'pra', 'pro', 'por', 'em', 'num', 'ao','aos','no', 'na', 'la',
                     'www','http','hhtps','fls','pg' }
    CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_numeric, strip_punctuation]
    CUSTOM_FILTERS_NUM = [lambda x: x.lower(), strip_tags, strip_punctuation]
    def preprocess_br(self, texto):
        # processa a lista de textos com threads
        if type(texto) is list or type(texto) is tuple:
           nova = [i for i in range(len(texto))]
           def _func(i):
               nova[i] = self.preprocess_br(texto[i])
           UtilDocs.map_thread(func = _func, lista = nova) 
           return nova
        
        # processa um texto
        texto = texto.replace('<br>', '\n')
        if self.config.token_br:
           texto = texto.replace('\n',' qbrpargf ')
        # caracteres que sobrevivem à limpeza
        texto = texto.replace('‘',' ').replace('“',' ').replace('”', ' ')
        if self.config.strip_numeric:
            tks = self.remove_stopwords(preprocess_string(texto, self.CUSTOM_FILTERS))
        else:
            tks = self.remove_stopwords(preprocess_string(texto, self.CUSTOM_FILTERS_NUM))
        # cria um token para a quebra de linha
        if self.config.token_br:
           tks = [_ if _ != 'qbrpargf' else '#br' for _ in tks]
        if not self.config.stemmer:
           return [unidecode(_) for _ in tks if len(_) >= 2]
        return [self.STEMMER.stem(_) for _ in tks if len(_) >= 2]        

    def remove_stopwords(self, tokens):
        return [_ for _ in tokens if _ and _ not in self.stop_words_usuario]

    def teste_modelo(self):
        #print('Init Sims', end='')
        #self.model.init_sims(replace=True)
        #print(' _o/')
        texto_1 = 'esse é um texto de teste para comparação - o teste depende de existirem os termos no vocab treinado'
        texto_2 = 'temos aqui um outro texto de teste para uma nova comparação mais distante - lembrando que o teste depende de existirem os termos no vocab treinado'
        texto_3 = 'esse é um texto de teste para comparação \n o teste depende de existirem os termos no vocab treinado'
        _texto_3 = texto_3.replace('\n',r'\n')
        texto_1_oov = texto_1.replace('texto','texto oovyyyyyyy oovxxxxxxxx').replace('teste', 'oovffffff teste oovssssss')
        UtilDocs.print_linha()
        print(' >>>> TESTE DO MODELO <<<<')
        UtilDocs.print_linha('- ')
        print('- Texto 01:  ', texto_1)
        print('  > tokens: ', '|'.join(self.preprocess_br(texto_1)) )
        print('- Texto 02:  ', texto_2)
        print('  > tokens: ', '|'.join(self.preprocess_br(texto_2)) )
        print('- Texto 03:  ', _texto_3)
        print('  > tokens: ', '|'.join(self.preprocess_br(texto_3)) )
        UtilDocs.print_linha('- ')
        sim = 100*self.similaridade(texto_1, texto_1)
        print(f'Similaridade entre o texto 01 e ele mesmo: {sim:.2f}%')          
        sim = 100*self.similaridade(texto_1, texto_1_oov)
        print(f'Similaridade entre o texto 01 e ele com oov: {sim:.2f}%')          
        sim = 100*self.similaridade(texto_1, texto_2)
        print(f'Similaridade entre os textos 01 e 02: {sim:.2f}%')          
        sim = 100*self.similaridade(texto_1, texto_3)
        print(f'Similaridade entre os textos 01 e 03: {sim:.2f}%')          
        UtilDocs.print_linha()

    # carrega os documentos txt de uma pasta e usa como tag o que estiver depois da palavra tags separadas por espaço
    # exemplo: 'arquivo texto 1 tags a b c.txt' tags será igual a ['a', 'b', 'c']
    def carregar_documentos(self, pasta):
        return UtilDocs.carregar_documentos(pasta)
   
    def get_parametros_modelo(self, model = None):
        _model = model if model else self.model
        if not _model:
            return {}
        return {c:v for c,v in _model.__dict__.items() if c[0] != '_' and type(v) in (str, int, float)}

    def vetorizar_dados(self, dados, epocas):
        def _vetorizar(i):
            if 'vetor' in dados[i] and type(dados[i]['vetor']) in (list, tuple):
               return
            texto = dados[i].get('texto', '')
            vetor = self.vetor(texto=texto, epochs=epocas, retornar_tokens = False)
            dados[i]['vetor'] = vetor
            dados[i]['hash'] = UtilDocs.hash(texto) if texto else 'vazio'
        UtilDocs.map_thread(_vetorizar, lista = range(len(dados)), n_threads=100)     
        UtilDocs.progress_bar(1,1,' vetorização Doc2Vec finalizada                                ')

class DocumentosDoArquivo():
    '''recebe o nome de um arquivo json contendo os documentos de treinamento.
       Espera-se:
       - uma chave texto e uma chave opcional tags
       - um json por linha do arquivo
       A leitura é feita em blocos para permitir treinamento com um volume muito grande de dados
       Como criar o arquivo json:
       1) Criar um Doc2VecRapido com as configurações de tokenização (config dele)
       2) Criar o DocumentosDoArquivo indicando o arquivo de saída e o obj Doc2VecRapido
       3) Incluir uma lista de dataframes com os dados ou incluir os textos e tags
       Exemplo:
         dv = Doc2VecRapido(pasta_modelo=PASTA_MODELO)        
         docs = DocumentosDoArquivo(arquivo_treino = './nom.json', dv)
         docs.incluir_textos(arquivos = ['./arq1.feather', './arq2.feather'], 
                             coluna_texto='texto_limpo', 
                             coluna_tags='categorias')    
       Pode-se testar se o arquito está íntegro para iteração no treinamento:
         docs = DocumentosDoArquivo(arquivo_treino = './nom.json')
         docs.testar()
       * df: pode ser o nome do arquivo .feather, pode ser o dataframe ou uma lista de dataframes.
       * o teste não precisa receber o obj Doc2VecRapido pois só vai iterar os dados
       Após gerar o arquivo, pode-se usar ele no treinamento.
       Exemplo: 
         dv = Doc2VecRapido(arq_tagged_docs = './nome.json')
         dv.treinar()
         dv.testar()

    '''
    def __init__(self, arquivo_treino=None, 
                 dv: Doc2VecRapido = None, 
                 qtd_linhas_bloco = 10000,
                 shuffle = True,
                 extra_tagged_docs = []):
        self.arquivo_treino = arquivo_treino
        self.__num_iteracao__ = -1
        self.__i__ = 0
        self.__bloco_atual__ = -1
        self.__qtd_blocos__ = 0
        self.__total_registros__ = -1
        self.qtd_linhas_bloco = qtd_linhas_bloco
        self.shuffle = shuffle
        # para incluir outros tagged_docs na iteração se necessário
        self.extra_tagged_docs = extra_tagged_docs
        self.reiniciar()
        if os.path.isfile(arquivo_treino):
           print(f'Arquivo de treino encontrado com {self.__len__()} registros, pronto para treinamento ou inclusão de novos textos') 
        else: 
           print('Arquivo de treino não encontrado, pronto para inclusão de novos textos') 
        self.dv = dv 
        self.time = time()

    @property
    def num_iteracao(self) -> int:
        return self.__num_iteracao__
    @property
    def bloco_atual(self) -> int:
        return self.__bloco_atual__
    @property
    def qtd_blocos(self) -> int:
        if self.__total_registros__ < 0:
           self.__len__() 
        return self.__qtd_blocos__
    @property
    def i(self) -> int:
        return self.__i__

    def reiniciar(self):
        self.__blocos_df__ = None
        self.__blocos_textos__ = None
        self.__num_iteracao__ += 1
        self.__i__ = -1
        self.__bloco_atual__ = -1

    # método individual de inclusão de documentos que pode ser usado em threads
    def incluir_texto(self, texto, tags = None):
        if self.dv is None:
           msg = 'incluir_texto: ATENÇÃO: é necessário passar um dv = Doc2VecRapido para que o processador use as configurações de tokenização dele'
           raise Exception(msg)
        
        if type(texto) is str:
           tokens = self.dv.preprocess_br(texto)
        elif  type(texto) in (list, set, tuple):
           tokens = list(texto)
        else: 
           msg = f'O tipo de texto informado não é válido: {type(texto)}. Esperado: str, list, set ou tuple'
           raise Exception(msg) 
        _tags = self.dv.processar_tags(tags=tags) if tags is not None else None
        with open(self.arquivo_treino,'a') as f:
             f.write(json.dumps({'tokens': tokens, 'tags': _tags}) + '\n')
        return
    
    def incluir_textos(self, df:Union[list, str, pd.DataFrame], coluna_texto:str=None, coluna_tokens:str= None, coluna_tags:str=None):
        ''' recebe um dataframe ou uma lista de dataframes ou um arquivo feather e cria os tagged_docs das colunas informadas
            processa o texto ou usa os tokens se a coluna de tokens for informada
        '''
        comp_msg = 'incluir_textos_data_frame: '
        if type(df) in [list,set, tuple]:
            # vazio, nada a fazer
            if len(df) == 0: return 
            ''' recebeu uma lista de dataframes ou uma lista de nome de arquivos'''
            print(f'Carregando {len(df)} dataframes')
            for _item in df:
                self.incluir_textos(_item, 
                                    coluna_texto=coluna_texto,
                                    coluna_tokens=coluna_tokens,
                                    coluna_tags=coluna_tags)   
            return
        if type(df) is str:
            print(f'Carregando dataframe do arquivo:', df)
            colunas = [coluna_texto, coluna_tokens, coluna_tags]
            colunas = [_ for _ in colunas if _]
            if not any(colunas):
               erro = f'{comp_msg}: é necessário informar pelo menos a coluna de texto do DataFrame'
            _df = pd.read_feather(df, columns=colunas)
            print(f'Criando dicionários dos {len(_df)} registros com tokens e tags ...')
            self.incluir_textos(_df, 
                                 coluna_texto=coluna_texto,
                                 coluna_tokens=coluna_tokens,
                                 coluna_tags=coluna_tags)
            print(f'Processamento do arquivo concluído com {len(_df)} registros _o/')
            del _df
            return
        erro = ''
        # dataframes    
        if type(df) is not pd.DataFrame:
           erro = f'incluir_textos: o objeto informado não é um pandas DataFrame: recebido {type(df)}'
        elif coluna_tokens and coluna_tokens not in df.columns:
           erro = f'incluir_textos: o DataFrame não possui a coluna de tokens "{coluna_tokens}"'
        elif coluna_texto and coluna_texto not in df.columns:
           erro = f'incluir_textos: o DataFrame não possui a coluna de texto "{coluna_texto}"'
        elif coluna_tags and coluna_tags not in df.columns:
           erro = f'incluir_textos: o DataFrame não possui a coluna de tags "{coluna_tags}"'
        elif not (coluna_texto or coluna_tokens):
           erro = f'incluir_textos: é necessário informar a coluna de textos ou a coluna de tokens do DataFrame'
        if erro: raise Exception(erro) 
        total = len(df)        
        def _processar(item):
            texto = item[coluna_texto] if coluna_texto else item[coluna_tokens]
            self.incluir_texto(texto = texto, tags = item[coluna_tags] if coluna_tags else None)
            UtilDocs.progresso_continuado(i = self.__total_registros__, total=total)
            self.__total_registros__ += 1
        df.apply(_processar, axis=1)      
        print()                          

    def __iter__(self):
        return self
    
    def __len__(self):
        if self.__total_registros__ < 0:
            with open(self.arquivo_treino) as f:
                 count = sum(1 for _ in f)
            self.__total_registros__ = count
            self.__qtd_blocos__ = (self.__total_registros__ + self.qtd_linhas_bloco - 1) // self.qtd_linhas_bloco
        return self.__total_registros__

    def __next__(self): # Python 2: def next(self)
        ''' vai navegar na lista de json carregando blocos em memória'''  
        self.__i__ += 1
        if self.__blocos_textos__ is None or len(self.__blocos_textos__) ==0:
           self.__proximo_bloco__() 
        linha = self.__blocos_textos__.pop(0) 
        if type(linha) is TaggedDocument:
           return linha 
        if linha.get('tags'):
            # a inclusão da tag -i é para não coincidir com as tags dos tagged_docs extras
            # que já serão processandos com a tag i da ordem deles
            linha['tags'].append(f'#{self.i}')
        doc = TaggedDocument(linha['tokens'], linha.get('tags'))
        return doc
    
    def __read_chunks__(self):
        return pd.read_json(self.arquivo_treino, 
                           lines=True, 
                           orient='records', 
                           chunksize = self.qtd_linhas_bloco) 

    def __proximo_bloco__(self):
        inicio = False
        if self.__blocos_df__ is None:
           self.__blocos_df__ = self.__read_chunks__()
           inicio = True
        self.__bloco_atual__ += 1 
        # enquanto tiver blocos, atualiza o bloco de textos
        # se acabarem os blocos, retorna o erro de fim da iteração
        try:
           df = next(self.__blocos_df__)
        except StopIteration:
           # prapara para começar novamente se chamar outra vez
           self.reiniciar()
           raise StopIteration
        self.__blocos_textos__= df.to_dict(orient='records')
        if inicio:
            # os extras podem ser dicts ou TaggedDocs
            self.__blocos_textos__ += self.extra_tagged_docs
        if self.shuffle:
           random.shuffle(self.__blocos_textos__)
        del df
        return True

    def testar(self):    
        print('Iterando arquivo para valiação: ')
        tokens = set()
        tags = set()
        UtilDocs.progresso_continuado(' - contando tokens e tags ... ')
        total = len(self)
        b_old = -1
        for i, dados in enumerate(self):
           if b_old != self.bloco_atual:
              print(f'   - bloco: {self.bloco_atual+1}/{self.qtd_blocos}')
              b_old = self.bloco_atual
           tokens.update(dados.words)
           tags.update(dados.tags)
           UtilDocs.progresso_continuado(i=i, total=total)
        print(' ... finalizado _o/')
        print(f' - Total de registros: {len(self)} com {len(tokens)} tokens e {len(tags)} tags')
        print(f' - Alguns tokens: ', ','.join(list(tokens)[:5]))
        print(f' - Algumas tags: ', ','.join(list(tags)[:5]))
        print(f' - Testando nova iteração:')
        docs_bloco = 0 
        for i, dados in enumerate(self):
            docs_bloco = len(self.__blocos_textos__)
            break
        if docs_bloco > 0 or len(self) == 0:
           print(f'   - nova iteração iniciada com {docs_bloco} documentos no bloco {self.bloco_atual}: ok  _o/')
        else: 
           print(f'   - ATENÇÃO: Nova iteração iniciada e nenhum documento no bloco: falha')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Pasta do modelo')
    parser.add_argument('-pasta', help='pasta para armazenamento ou carregamento do modelo - padrao meu_modelo', required=False)
    parser.add_argument('-textos', help='pasta com textos para treinamento do modelo - padrao meus_textos', required=False)
    parser.add_argument('-epocas', help='número de épocas para treinamento - padrão 200 ', required=False)
    parser.add_argument('-config', help='gera o arquivo de config se não existir ', required=False, action='store_const', const=1)
    parser.add_argument('-skipgram', help='treina o modelo Skip gram ', required=False, action='store_const', const=1)
    parser.add_argument('-skip_gram', help='treina o modelo Skip gram ', required=False, action='store_const', const=1)
    args = parser.parse_args()

    config = args.config
    skip_gram = True if args.skipgram or args.skip_gram else None

    PASTA_MODELO = str(args.pasta) if args.pasta else './meu_modelo'
    PASTA_TEXTOS = os.path.realpath(args.textos) if args.textos else None
    if not PASTA_TEXTOS and args.textos and str(args.textos).endswith('/'):
       PASTA_TEXTOS = os.path.basename(args.textos[:-1])
    elif not PASTA_TEXTOS and args.textos and str(args.textos).endswith('\\'):
       PASTA_TEXTOS = os.path.basename(args.textos[:-1])
    EPOCAS = int(args.epocas) if args.epocas else 200

    # sem parâmetros, faz apenas o teste
    if not PASTA_TEXTOS:
       print('############################################################')
       print('# Nenhuma pasta de texto informada - testando o modelo     #') 
       print('# Use -h para verificar os parâmetros disponíveis          #')
       print('#----------------------------------------------------------#')
       if Doc2VecRapido.modelo_existe(PASTA_MODELO):
           print(f'- modelo encontrado em "{PASTA_MODELO}"')
       else:
           if config:
              Doc2VecRapido.novo_config(PASTA_MODELO)
              exit()  
           print(f' ERRO: modelo não encontrado em "{PASTA_MODELO}"')
           exit()
       dv = Doc2VecRapido(pasta_modelo=PASTA_MODELO)
       dv.teste_modelo()
       if config:
          print('Config do modelo: ', dv.config)
       exit()

    if not os.path.isdir(PASTA_TEXTOS):
       print(f'ERRO: Não foi encontrada a pasta com os textos para treinamento em "{PASTA_TEXTOS}" ')

    # treinamento
    dv = Doc2VecRapido(pasta_modelo=PASTA_MODELO, documentos=PASTA_TEXTOS, epochs=EPOCAS, skip_gram = skip_gram)
    if config:
       print('Config do modelo: ', dv.config)
    else:
       dv.treinar()
       dv.teste_modelo()

