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

'''
  config.json = {"vector_size": 300, "strip_numeric":true, "stemmer":false, "min_count": 5 , "token_br": true}

  strip_numeric = remove números
  stemmer = utiliza o stemmer dos tokens
  min_count = ocorrência mínima do token no corpus para ser treinado
  token_br = cria o token #br para treinar simulando a quebra de parágrafos
  vector_size = número de dimensões do vetor que será treinado
'''
#####################################
from multiprocessing.dummy import Pool as ThreadPool
def map_thread(func, lista, n_threads=5):
    # print('Iniciando {} threads'.format(n_threads))
    pool = ThreadPool(n_threads)
    pool.map(func, lista)
    pool.close()
    pool.join()  

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

    def __init__(self, pasta_modelo = PASTA_PADRAO, documentos = [], tags = [], 
                       vector_size = 300, 
                       min_count = 5, 
                       epochs = 1000,
                       strip_numeric = True,
                       stemmer = False,
                       skip_gram = False) -> None:
        # configura arquivos da classe
        self.pasta_modelo = os.path.basename(pasta_modelo)
        self.arquivo_config = os.path.join(self.pasta_modelo,'config.json')
        self.arquivo_modelo = os.path.join(pasta_modelo, self.ARQUIVO_MODELO)
        self.arquivo_vocab = os.path.join(pasta_modelo,'vocab_treinado.txt')
        self.arquivo_vocab_sim = os.path.join(pasta_modelo,'vocab_similares.txt')
        self.arquivo_stopwords = os.path.join(pasta_modelo,'stopwords.txt')
        # carrega as configurações se existirem
        self.config = Config(arquivo=self.arquivo_config,
                             min_count = min_count, 
                             vector_size=vector_size, 
                             strip_numeric=strip_numeric, 
                             stemmer=stemmer,
                             skip_gram=skip_gram)
        self.model = None
        self.documentos_carregados = 0
        self.tagged_docs = None 
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
            if not any(documentos):
               return
        if (not os.path.isfile(self.arquivo_modelo)) and (not any(documentos)):
           raise Exception(self.printlog(f'não foi possível encontrar o modelo "{self.arquivo_modelo}" ou documentos para treiná-lo',True))
        # verificar e criar a pasta para o modelo
        os.makedirs(self.pasta_modelo, exist_ok=True)
        self.min_count = self.config.min_count
        self.epochs = epochs
        # caso documentos seja uma string, verifica se é uma pasta com documentos
        if type(documentos) is str and os.path.isdir(documentos):
           _docs, _tags = self.carregar_documentos(documentos)
           self.tagged_docs = self.pre_processar(documentos =_docs, tags = _tags, retornar_tagged_doc = True)
        else:
            self.tagged_docs = self.pre_processar(documentos =documentos, tags = tags, retornar_tagged_doc = True)
        self.documentos_carregados = len(documentos)
        self.printlog(f'Pronto para treinar o modelo {self.pasta_modelo} com {self.documentos_carregados} documentos')
        self.config.gravar_config(self.arquivo_config, self.get_parametros_modelo())

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

    def printlog(self,msg, retornar_msg_erro=False, destaque = False):
        msg = f'> Doc2VecRapido: {msg}'
        if retornar_msg_erro:
            return msg
        if destaque: 
           self.print_linha()
        print(msg)
        if destaque: 
           self.print_linha()

    def print_linha(self, caractere='='):
        print(str(caractere * 70)[:70])

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
                   res.append( TaggedDocument(tokens, str(tags[i])) )
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
        self.printlog(f'Treinando modelo com {self.documentos_carregados} documentos e {self.epochs} épocas \n - CONFIG: {self.config}', destaque=True)
        if self.model is None:
           self.config.skip_gram_window = max(1, self.config.skip_gram_window)
           self.config.window = max(1, self.config.window)
           self.config.vector_size = max(10, self.config.vector_size)
           self.config.min_count = max(1, self.config.min_count)
           model = Doc2Vec(vector_size=self.config.vector_size, 
                           min_count=self.min_count, 
                           epochs=self.epochs,
                           window = self.config.skip_gram_window if self.config.skip_gram else self.config.window)
           if self.config.skip_gram:
              model.sg = 1 
           self.printlog(f'Criando vocab para o novo modelo')
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
              model.train(self.tagged_docs, total_examples=model.corpus_count, epochs=bloco_epocas)
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

        self.printlog(f'Modelo treinado com {self.config.log_treino_vocab} termos e {self.config.vector_size} dimensões')
        self.printlog(f'Modelo treinado com {self.epochs} épocas, totalizando {self.config.log_treino_epocas} desde sua criação.')
        if self.config.log_treino_epocas >= self.config.max_total_epocas:
           self.printlog(f'Treinamento atingiu o máximo de épocas configuradas: "{self.config.max_total_epocas}"')
        self.printlog(f'Modelo gravado em "{self.arquivo_modelo}"', destaque=True)


    def vetor(self, texto, epochs = 100, normalizar = True, retornar_tokens = False):
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
           map_thread(func = _func, lista = nova) 
           return nova
        
        # processa um texto
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
           return [_ for _ in tks if len(_) >= 2]
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
        self.print_linha()
        print(' >>>> TESTE DO MODELO <<<<')
        self.print_linha('- ')
        print('- Texto 01:  ', texto_1)
        print('  > tokens: ', '|'.join(self.preprocess_br(texto_1)) )
        print('- Texto 02:  ', texto_2)
        print('  > tokens: ', '|'.join(self.preprocess_br(texto_2)) )
        print('- Texto 03:  ', _texto_3)
        print('  > tokens: ', '|'.join(self.preprocess_br(texto_3)) )
        self.print_linha('- ')
        sim = 100*self.similaridade(texto_1, texto_1)
        print(f'Similaridade entre o texto 01 e ele mesmo: {sim:.2f}%')          
        sim = 100*self.similaridade(texto_1, texto_1_oov)
        print(f'Similaridade entre o texto 01 e ele com oov: {sim:.2f}%')          
        sim = 100*self.similaridade(texto_1, texto_2)
        print(f'Similaridade entre os textos 01 e 02: {sim:.2f}%')          
        sim = 100*self.similaridade(texto_1, texto_3)
        print(f'Similaridade entre os textos 01 e 03: {sim:.2f}%')          
        self.print_linha()

    # carrega os documentos txt de uma pasta e usa como tag o que estiver depois da palavra tags separadas por espaço
    # exemplo: 'arquivo texto 1 tags a b c.txt' tags será igual a ['a', 'b', 'c']
    def carregar_documentos(self, pasta):
        self.printlog(f'Carregando documentos da pasta "{pasta}"')
        arquivos = self.listar_arquivos(pasta)
        self.printlog(f'Documentos encontrados: {len(arquivos)}')
        documentos = []
        def _func(i):
            arquivo = arquivos[i]
            documento = self.carregar_arquivo(arquivo)
            rotulo = os.path.split(arquivo)[1]            
            rotulo = os.path.splitext(rotulo)[0].lower()
            # se encontrar tag ou tags no nome do arquivo, busca as tags
            if rotulo.find('tags ') >=0:
               rotulos = rotulo.split('tags ')[1].strip().split(' ')
               print(f'Rótulo encontrado: {arquivo} >> tags: {rotulos}')
            elif rotulo.find('tag ') >=0:
               rotulos = rotulo.split('tag ')[1].strip().split(' ')
               print(f'Rótulo encontrado: {arquivo} >> tags: {rotulos}')
            else:
               rotulos = [rotulo.strip()]
            documentos.append( (documento, rotulos) )
        map_thread(_func,list(range(len(arquivos))) )
        self.printlog(f'Documentos carregados: {len(documentos)}')
        #documentos, rotulos = list(zip(*documentos))
        #self.printlog(f'Exemplo de documento: {documentos[0]} e rótulos: {rotulos[0]}')
        return list(zip(*documentos))

    # função simples de carga de arquivos que tenta descobrir o tipo de arquivo (utf8, ascii, latin1)
    @classmethod
    def carregar_arquivo(cls, arquivo):
        tipos = ['utf8', 'ascii', 'latin1']
        texto = None
        for tp in tipos:
            try:
                with open(arquivo, encoding=tp) as f:
                    texto = f.read()
                    break
            except UnicodeError:
                continue
        if not texto:
            with open(arquivo, encoding='latin1', errors='ignore') as f:
                texto = f.read()
        return texto.replace('<br>','\n') 

    @classmethod
    def listar_arquivos(cls, pasta, extensao='txt'):
        if not os.path.isdir(pasta):
            msg = f'Não foi encontrada a pasta "{pasta}" para listar os arquivos "{extensao}"'
            raise Exception(msg)
        res = []
        _extensao = f".{extensao}".lower() if extensao else ''
        for path, _, file_list in os.walk(pasta):
            for file_name in file_list:
                if file_name.lower().endswith(f"{_extensao}"):
                   res.append(os.path.join(path,file_name))
        return res
    
    def get_parametros_modelo(self, model = None):
        _model = model if model else self.model
        if not _model:
            return {}
        return {c:v for c,v in _model.__dict__.items() if c[0] != '_' and type(v) in (str, int, float)}

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

    PASTA_MODELO = os.path.basename(args.pasta) if args.pasta else './meu_modelo'
    PASTA_TEXTOS = os.path.basename(args.textos) if args.textos else None
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

