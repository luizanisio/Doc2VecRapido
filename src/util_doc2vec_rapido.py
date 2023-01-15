# -*- coding: utf-8 -*-

#######################################################################
# Classes: 
# Doc2VecRapido : permite carregar um modelo Doc2Vec treinado para aplicar em documentos de uma determinata área de conhecimento
#                 permite ajustar as configurações do treinamento usando o arquivo config.json
# Esse código, dicas de uso e outras informações: 
#   -> https://github.com/luizanisio/Doc2VecRapido
# Luiz Anísio 
# 14/01/2023 - disponibilizado no GitHub  
#######################################################################

import logging

import os
import json

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from numpy import linalg
from scipy.spatial import distance

from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_numeric
from gensim.corpora.textcorpus import remove_stopwords
from nltk.stem.snowball import SnowballStemmer

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
                 vector_size = 300, min_count = 5)  -> None:
        self.strip_numeric = True
        self.stemmer = False
        self.vector_size = 300
        self.min_count = 5
        self.window = 10
        self.max_total_epocas = 0
        self.token_br = True
        self.log_treino_epocas = 0
        self.log_treino_vocab = 0
        if arquivo:
           self.carregar(arquivo) 
    
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
                   self.max_total_epocas = _config.get('max_total_epocas',self.max_total_epocas)
                   self.log_treino_epocas = _config.get('log_treino_epocas',0)
                   self.log_treino_vocab = _config.get('log_treino_vocab',0)
                   self.print(f'> Config: carregado: {self.config}')
                except:
                   return

    def gravar_config(self, arquivo):
        with open(arquivo, 'w') as f:
            f.write(json.dumps(self.as_dict(), indent=2))

class Doc2VecRapido():
    CORT_SIM_VOCAB_EXEMPLOS = 0.7 # corte para gereação do arquivo de exemplos de similaridade

    def __init__(self, pasta_modelo = 'd2vmodel', documentos = [], tags = [], 
                       vector_size = 300, 
                       min_count = 5, 
                       epochs = 1000,
                       strip_numeric = True,
                       stemmer = False) -> None:
        # configura arquivos da classe
        self.pasta_modelo = os.path.basename(pasta_modelo)
        self.arquivo_config = os.path.join(self.pasta_modelo,'config.json')
        self.arquivo_modelo = os.path.join(pasta_modelo,'doc2vecrapido.d2v')
        self.arquivo_vocab = os.path.join(pasta_modelo,'vocab_treinado.txt')
        self.arquivo_vocab_sim = os.path.join(pasta_modelo,'vocab_similares.txt')
        self.arquivo_stopwords = os.path.join(pasta_modelo,'stopwords.txt')
        # carrega as configurações se existirem
        self.config = Config(arquivo=self.arquivo_config,
                             min_count = min_count, 
                             vector_size=vector_size, 
                             strip_numeric=strip_numeric, 
                             stemmer=stemmer)
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
            # atualiza o config se precisar pois alguns parâmetros não mudam depois do treinamento
            # corrige o config para ficar consistente com o modelo
            if (self.config.vector_size != self.model.wv.vector_size) or \
               (self.config.log_treino_vocab != len(self.model.wv.vocab) ) or \
               (self.config.window != self.model.window) :  
                self.config.vector_size = self.model.wv.vector_size
                self.config.window = self.model.window
                self.config.log_treino_vocab = len(self.model.wv.vocab)
                self.config.gravar_config(self.arquivo_config)
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
        self.config.gravar_config(self.arquivo_config)

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

    def treinar(self):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        self.printlog(f'Treinando modelo com {self.documentos_carregados} documentos e {self.epochs} épocas \n - CONFIG: {self.config}', destaque=True)
        if self.model is None:
           model = Doc2Vec(vector_size=self.config.vector_size, 
                           min_count=self.min_count, 
                           epochs=self.epochs,
                           window = self.config.window)
           self.printlog(f'Criando vocab para o novo modelo')
           model.build_vocab(self.tagged_docs)
           self.printlog(f'Treinando novo modelo')
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
        while total_epocas > 0:
              bloco_epocas = 50 if total_epocas > 50 else total_epocas
              total_epocas -= bloco_epocas
              model.train(self.tagged_docs, total_examples=model.corpus_count, epochs=bloco_epocas)
              epocas_treinadas += bloco_epocas
              self.printlog(f'Gravando modelo após bloco com {bloco_epocas} épocas\n - total {epocas_treinadas} épocas treinadas \n - épocas restantes: {total_epocas}', destaque=True)
              self.config.log_treino_epocas = epocas_treinadas
              self.config.log_treino_vocab = len(model.wv.vocab)
              model.save(self.arquivo_modelo)
              self.config.gravar_config(self.arquivo_config) # atualiza o config
        self.model = model 
        self.printlog(f'Gravando vocab treinado')
        # vocab treinado
        with open(self.arquivo_vocab,'w') as f:
             f.write('\n'.join(sorted([str(_) for _ in self.model.wv.vocab])))
        # similares do vocab
        self.printlog(f'Gravando log de similaridade entre os termos')
        with open(self.arquivo_vocab_sim,'w') as f:
             for termo in sorted(list(self.model.wv.vocab)):
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


    def vetor(self, texto, epochs = 100, normalizar = True):
        tokens = self.pre_processar(documentos = texto, retornar_tagged_doc=False)
        vetor = self.model.infer_vector(tokens, epochs = epochs) 
        if normalizar:
           return [float(f) for f in vetor / linalg.norm(vetor)]     
        return [float(f) for f in vetor]

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
        if self.config.strip_numeric:
            tks = remove_stopwords(preprocess_string(texto, self.CUSTOM_FILTERS), stopwords = self.stop_words_usuario)
        else:
            tks = remove_stopwords(preprocess_string(texto, self.CUSTOM_FILTERS_NUM), stopwords= self.stop_words_usuario)
        # cria um token para a quebra de linha
        if self.config.token_br:
           tks = [_ if _ != 'qbrpargf' else '#br' for _ in tks]
        if not self.config.stemmer:
           return tks
        return [self.STEMMER.stem(_) for _ in tks]        

    def teste_modelo(self):
        texto_1 = 'esse é um texto de teste para comparação - o teste depende de existirem os termos no vocab treinado'
        texto_2 = 'esse outro texto de teste para uma nova comparação - lembrando que o teste depende de existirem os termos no vocab treinado'
        texto_3 = 'esse é um texto de teste para comparação \n o teste depende de existirem os termos no vocab treinado'
        _texto_3 = texto_3.replace('\n',r'\n')
        texto_1_oov = texto_1.replace('texto','texto oovyyyyyyy oovxxxxxxxx').replace('teste', 'oovffffff teste oovssssss')
        self.print_linha()
        print(' >>>> TESTE DO MODELO <<<<')
        self.print_linha('- ')
        print('Texto 1: ', texto_1)
        print('Texto 2: ', texto_2)
        print('Texto 3: ', _texto_3)
        self.print_linha('- ')
        sim = 100*self.similaridade(texto_1, texto_1)
        print(f'Similaridade entre o texto 1 e ele mesmo: {sim:.2f}%')          
        sim = 100*self.similaridade(texto_1, texto_1_oov)
        print(f'Similaridade entre o texto 1 e ele com oov: {sim:.2f}%')          
        sim = 100*self.similaridade(texto_1, texto_2)
        print(f'Similaridade entre os textos 1 e 2: {sim:.2f}%')          
        sim = 100*self.similaridade(texto_1, texto_3)
        print(f'Similaridade entre os textos 1 e 3: {sim:.2f}%')          
        self.print_linha()

    # carrega os documentos txt de uma pasta e usa como tag o que estiver depois da palavra tags separadas por espaço
    # exemplo: 'arquivo texto 1 tags a b c.txt' tags será igual a ['a', 'b', 'c']
    def carregar_documentos(self, pasta):
        self.printlog(f'Carregando documentos da pasta "{pasta}"')
        arquivos = self.listar_arquivos(pasta)
        self.printlog(f'Documentos encontrados: "{len(arquivos)}"')
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
        return texto

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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Pasta do modelo')
    parser.add_argument('-pasta', help='pasta para armazenamento ou carregamento do modelo - padrao meu_modelo', required=False)
    parser.add_argument('-textos', help='pasta com textos para treinamento do modelo - padrao meus_textos', required=False)
    parser.add_argument('-epocas', help='número de épocas para treinamento - padrão 200 ', required=False)
    args = parser.parse_args()

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
           print(f' ERRO: modelo não encontrado em "{PASTA_MODELO}"')
           exit()
       dv = Doc2VecRapido(pasta_modelo=PASTA_MODELO)
       dv.teste_modelo()
       exit()

    if not os.path.isdir(PASTA_TEXTOS):
       print(f'ERRO: Não foi encontrada a pasta com os textos para treinamento em "{PASTA_TEXTOS}" ')

    # treinamento
    dv = Doc2VecRapido(pasta_modelo=PASTA_MODELO, documentos=PASTA_TEXTOS, epochs=EPOCAS)
    dv.treinar()
    dv.teste_modelo()

