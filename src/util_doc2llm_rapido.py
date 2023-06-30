# -*- coding: utf-8 -*-
#######################################################################
# Código para vetorizar documentos usando LLMs
#
# Esse código, dicas de uso e outras informações: 
#   -> https://github.com/luizanisio/Doc2VecRapido/
# Luiz Anísio 
# 09/06/2023 - disponibilizado no GitHub  
#######################################################################

from sentence_transformers import SentenceTransformer
from scipy.spatial import distance
from torch.nn import DataParallel
from torch.cuda import device_count
from util_doc2util import UtilDocs, Progresso
from numpy import linalg
import warnings

import os
import json

class Doc2LLMRapido():
    # https://huggingface.co/tgsc/sentence-transformer-ult5-pt-small
    # 215Mb 256 entrada - 768 saída - 110Mi param
    T5 = 'sentence-transformers/sentence-t5-base'
    # 650Mb - 256 entrada - 768 saída - 335Mi param
    T5L = 'sentence-transformers/sentence-t5-large'
    # 2.3Gb 6GbRAM - 512 entrada - 768 saída - 1.2Bi param
    T5XL = 'sentence-transformers/sentence-t5-xl'
    # 10Gb - 512 entrada - 768 saída - 11Bi param
    T5XXL = 'sentence-transformers/sentence-t5-xxl'  

    # 200Mb 1024 entrada - 512 saída - 51Mi param
    T5BR = 'tgsc/sentence-transformer-ult5-pt-small' 
    # 256Mb 512 entrada - 768 saída - 66Mi param
    DISTILBERT = 'sentence-transformers/multi-qa-distilbert-cos-v1'

    # OS MELHORES  
    # semântica AQ 2.3Gb - 512 entrada - 768 saída - 4.8Bi parâmetros
    GTRT5XL = 'sentence-transformers/gtr-t5-xl'
    # semântica 9.1 Gb - 512 entrada - 768 saída - 11Bi param
    GTRT5XXL = 'sentence-transformers/gtr-t5-xxl'

    # 2.5 Gb 512 entrada - 1024 saída - 334Mi param
    BERT = 'neuralmind/bert-large-portuguese-cased'
    BERT_LARGE = 'neuralmind/bert-large-portuguese-cased'
    # 1.3 Gb 512 entrada - 768 saída - 109Mi param
    BERT_BASE = 'neuralmind/bert-base-portuguese-cased'
    
    # BERT 4k - ocorre erro de cálculo de tokens pois considera 4098 tokens no encode 
    # - é necessário corrigir o arquivo sentence_bert_config.json para 4096 tokens
    # 600Mb 4096 tokens e entrada - 768 dimensões de saída - 148Mi param
    BERT_4K = 'allenai/longformer-base-4096'

    # estratégia de vetorização de textos longos
    LONGOS_MEDIA = 'média'
    LONGOS_SOMA = 'soma'
    LONGOS_NENHUMA = 'nenhuma' # fará o truncamento do texto
    MEDIA_CHAR_TOKEN = 3.5 # para previsão do número de tokens na criação dos chunks dos textos
    def __init__(self, modelo = BERT, path = '', device=None, ignore_warnings=True):
      ''' modelo = pode ser o nome do modelo remoto, uma das constantes dessa classe ou o caminho do modelo
          path = pasta onde estão os modelos locais para não precisar baixar do endereço remoto
                 caso não encontre o modelo, vai tentar baixar remoto e colocar nessa pasta
          max_length = None busca automaticamente do modelo
          device = auto ou None -: identifica automaticamente'''
      _device = 'auto' if not device else device
      self.warnings = warnings
      #print(f'Iniciando modelo: {modelo} (device: {device} | GPUs : {device_count()})')    

      # verifica se o modelo passado é uma das constantes
      modelo_busca = self.get_model_cst(modelo)
      modelo_busca = modelo_busca if modelo_busca else modelo
      # possibilidades de localização do modelo  
      pastas = ['./','./llms/','./modelos/','./models/', str(path)]
      nomes = [modelo_busca]
      nomes.append( modelo_busca.split('/')[-1] if modelo_busca.find('/') >=0 else modelo_busca )
      # não existe a pasta do modelo, busca nas possíveis pastas
      if os.path.isdir(modelo_busca):
          _modelo = str(modelo_busca)
      else:
          _modelo = ''
          for pasta in pastas:
              if not pasta:continue  
              for nome in nomes: 
                  if not nome:continue  
                  _teste = os.path.join(pasta, nome)  
                  if os.path.isdir(_teste):
                     _modelo = _teste
                     break
              if _modelo:
                 break
      if not _modelo:
         # vai baixar do repositório remoto
         if path:
            os.makedirs(path, exist_ok=True)
            self.printlog(f'Baixando o modelo para "{path}": {modelo_busca}')   
            #self.model = SentenceTransformer(modelo, cache_folder=path, warnings=self.warnings)
            self.nome_modelo = modelo_busca
         else:
            self.printlog(f'Baixando o modelo: {modelo_busca}')   
            #self.model = SentenceTransformer(modelo, warnings=self.warnings)
            self.nome_modelo = modelo_busca
      else:
         self.printlog(f'Iniciando modelo: {_modelo}')      
         #self.model = SentenceTransformer(_modelo, warnings=self.warnings)
         self.nome_modelo = _modelo
      # temos GPU e é para usar GPU ou auto
      _to_cuda = False
      if _device.lower() == 'auto':
          if device_count() >= 1:
             _to_cuda = True
             self.printlog(' - Vamos tentar usar GPU...  _o/')
             _device = 'cuda'
          else:
             _device = 'cpu'
      elif device.lower() != 'cpu':
          if device_count() == 0:
             self.printlog(' - Estamos sem GPU, vamos usar CPU...  /o\\')
             _device = 'cpu'

      self.device = _device
      self.printlog(f' - Carregando modelo "{self.nome_modelo}" para {self.device}')
      if ignore_warnings:
         warnings.filterwarnings("ignore") 
      self.model = SentenceTransformer(self.nome_modelo, cache_folder = path, device = self.device)
      if ignore_warnings:
         warnings.filterwarnings("default") 
      # tamanho máximo de tokens do modelo
      self.max_length = self.get_max_seq_length()
      # parâmetros para chunks de textos longos
      self.get_parametros_chunks()
      if _to_cuda:
         self.device = 'cuda'
         self.printlog('Enviando modelo para DataParallel com GPU ...')
         self.model = DataParallel(self.model)
         self.model.to(self.device)
      self.printlog(f'Modelo "{self.nome_modelo}" pronto para uso (device = {self.device} | GPUs : {device_count()}) _/o')

    def get_model(self):
        return self.model.module if type(self.model) is DataParallel else self.model
    
    def get_model_cst(self, atributo: str):
        _atributo  = str(atributo).upper()
        if hasattr(self, _atributo):
            valor = getattr(self, _atributo)
            if type(valor) is str:
               return valor
        return None
    
    def get_config_max_seq_length(self):
        return self.config.get('max_seq_length', self.config.get('max_length'))

    def get_max_seq_length(self):
        _model = self.get_model()
        max_limit = 100000
        if hasattr(_model, 'get_max_seq_length'):
            max_length = _model.get_max_seq_length()
            if max_length is not None and 0 < max_length < max_limit:
               return max_length
        if hasattr(_model, 'tokenizer') and hasattr(_model.tokenizer, 'max_len'):
            max_length = _model.tokenizer.max_len
            if max_length is not None and 0 < max_length < max_limit:
               return max_length
        if hasattr(_model, 'tokenizer') and hasattr(_model.tokenizer, 'model_max_length'):
            max_length = _model.tokenizer.model_max_length
            if max_length is not None and 0 < max_length < max_limit:
               return max_length
        self.printlog(" - não foi possível determinar o número máximo de tokens de entrada para o modelo, assumindo self.max_length=512, altere se necessário")    
        return 512

    def get_special_tokens(self):
        special = []
        def _get_item_(item):
            if type(item) is list:
                for _ in item:
                    _get_item_(_)
            else:
               if item.find('extra_id') == -1: 
                  special.append(item)  
        _get_item_(list(self.get_model().tokenizer.special_tokens_map.values()))
        return special

    def info(self):
        self.printlog(f'> Maior sequência de entrada: {self.max_length} tokens')
        self.printlog(f'> Dimensões do vetor de saída: {len(self.vetor("teste"))}')
        params = int(sum(p.numel() for p in self.model.parameters())/1000)
        params = f'{params}k' if params < 1000 else f'{int(params/1000)}Mi'
        self.printlog(f'> Parâmetros do modelo: {params}')
        self.printlog(f'> Special tokens: {self.get_special_tokens()}') 
        self.printlog(f'> Mask token: {self.get_model().tokenizer.mask_token}')

    def similaridade(self, texto1, texto2, estrategia_longos = LONGOS_MEDIA):
        ''' similaridade entre dois textos
        '''
        vetor1, vetor2 = self.vetores([texto1, texto2], estrategia_longos=estrategia_longos, progresso = False) 
        return 1- distance.cosine(vetor1,vetor2)   

    def similaridade_pares(self, pares_textos, estrategia_longos = LONGOS_MEDIA):
        ''' recebe uma lista de pares [(texto1, texto2), ....]
            retorna uma lista de similaridades entre os pares
        '''
        textos1, textos2 = zip(*pares_textos)
        if len(pares_textos) == 0:
            return []
        if len(pares_textos) > 1:
           self.printlog(f'> Vetorizando {len(textos1)} textos 1 da comparação ...')
        vetores1 = self.vetorizar_textos(textos1, estrategia_longos=estrategia_longos, progresso = True)
        if len(pares_textos) > 1:
           self.printlog(f'> Vetorizando {len(textos2)} textos 2 da comparação ...')
        vetores2 = self.vetorizar_textos(textos2, estrategia_longos=estrategia_longos, progresso = True)
        sims = UtilDocs.map_thread(self.similaridade_par_vetores, list(zip(vetores1, vetores2)) )
        return sims

    def similaridade_vetores(self, vetor1, vetor2):
        ''' recebe dois vetores e retorna a similaridade entre eles (1-distância) '''
        return 1- distance.cosine(vetor1,vetor2) 

    def similaridade_par_vetores(self, par_vetores):
        ''' recebe um par de vetores
            retorna a similaridades entre o par
        '''
        return self.similaridade_vetores(*par_vetores)  

    def vetor(self, texto, retornar_float = False, normalizar = True, batch_size=8, estrategia_longos = LONGOS_MEDIA, progresso=False):
        return self.vetores(textos = [texto], retornar_float = retornar_float, normalizar = normalizar, batch_size=batch_size, estrategia_longos = estrategia_longos, progresso=progresso)[0]
    
    def e_texto_longo(self, texto):
        return len(texto) > self.__parametros_chunks__['nao_longo']
    
    def get_parametros_chunks(self,):
        tamanho_maximo = int(self.MEDIA_CHAR_TOKEN * self.max_length * 0.7)
        tolerancia = int(tamanho_maximo * 0.15)
        sobreposicao = int(tamanho_maximo / 3)
        self.__parametros_chunks__ = {'maximo' : tamanho_maximo - tolerancia, 
                                      'tolerancia' : tolerancia,
                                      'sobreposicao' : sobreposicao,
                                      'nao_longo' : tamanho_maximo}

    # vetorização de textos curtos e textos longos com a estratégia definida
    def vetores(self, textos, retornar_float = False, normalizar = True, batch_size=8, estrategia_longos = LONGOS_MEDIA, progresso=True):
        _model = self.get_model()
        vetores = [] # guarda o índice e o vetor
        textos_grandes = []
        indices_textos_grandes = []
        textos_pequenos = []
        indices_textos_pequenos = []
        # chunks - o máximo com a tolerância é o limite do modelo
        tamanho_maximo = self.__parametros_chunks__['maximo']
        sobreposicao = self.__parametros_chunks__['sobreposicao']
        tolerancia = self.__parametros_chunks__['tolerancia']
        nao_longo = self.__parametros_chunks__['nao_longo']

        if estrategia_longos == self.LONGOS_NENHUMA:
            textos_pequenos = textos # todos os textos serão truncados se passarem do limite
        else:
            for i, texto in enumerate(textos):
                if len(texto) > nao_longo:
                    textos_grandes.append(texto)
                    indices_textos_grandes.append(i)
                else:
                    textos_pequenos.append(texto)
                    indices_textos_pequenos.append(i)

        if len(textos_grandes) > 0:
            bar = Progresso(len(textos_grandes)) if progresso else None
            for texto, indice in zip(textos_grandes, indices_textos_grandes):
                if bar: 
                   bar.update()
                chunks = UtilDocs.quebrar_pedacos(texto = texto, tamanho=tamanho_maximo, sobreposicao=sobreposicao, tolerancia=tolerancia, incluir_trecho = True)
                chunks = [_['trecho'] for _ in chunks]
                if len(chunks) == 0:
                    chunks = [texto]
                # vetoriza em nparray para consolidar os vetores resultantes 
                vetor_pooling = self.vetores(textos = chunks, retornar_float = False, batch_size=batch_size, estrategia_longos=self.LONGOS_NENHUMA, progresso = False)
                #print('Vetor longo ok', indice, 'consolidando')
                vetores.append((indice, self.consolidar_vetores(vetor_pooling, estrategia = estrategia_longos, normalizar=normalizar) ))
                #print('Vetor longo ok', indice)
            if bar: 
               bar.close()

        # caso só tenha textos pequenos, não perde tempo reordenando pelo índice
        vetores_finais = None
        if len(textos_pequenos) > 0:
            vetores_textos_pequenos = _model.encode(textos_pequenos, normalize_embeddings=normalizar, show_progress_bar = (progresso and len(textos_pequenos)>1), device=self.device)
            if len(textos_grandes) == 0:
                vetores_finais = vetores_textos_pequenos
            else:
                for vetor, indice in zip(vetores_textos_pequenos, indices_textos_pequenos):
                    vetores.append((indice, vetor))

        # ordena os textos pequenos e grandes com a ordem original
        if vetores_finais is None:
            vetores_ordenados = sorted(vetores, key=lambda x: x[0])
            vetores_finais = [vetor for _, vetor in vetores_ordenados]

        if retornar_float:
           return [_.tolist() for _ in vetores_finais] 
        return vetores_finais
  
    def consolidar_vetores(self, vetores, estrategia = LONGOS_MEDIA, normalizar = True):
        '''Consolida vetores de textos longos'''
        if estrategia == self.LONGOS_MEDIA:
            res = sum(vetores) / len(vetores)
        else:
            res = sum(vetores)
        if normalizar:
           return res / linalg.norm(res)     
        return res  

    def vetorizar_dados(self, dados, epocas = None):
         _pos = []
         _textos = []
         # verifica dados que não possuem vetor (pode ter vindo do cache ou trazidos da base de dados)
         self.printlog(f'Analisando {len(dados)} textos ...')
         #qtd = len(dados)
         qtd_vazios, qtd_ok = [0], [0]
         bar = Progresso(total = len(dados))
         restam = [len(dados)]
         #for i in range(len(dados)):
         def __pre_avaliar__(i):
             #UtilDocs.progress_bar(i, qtd,f' vetorizando textos grandes {i+1}/{qtd} textos                ')
             restam[0] -= 1
             bar.restantes(restam[0])
             if 'vetor' in dados[i] and type(dados[i]['vetor']) in (list, tuple):
                # já tem vetor, verifica o hash ou cria um único
                qtd_ok[0] += 1
                return
             
             _texto = dados[i].get('texto','')
             # sem texto
             if not _texto:
                dados[i]['texto'] = ''
                dados[i]['hash'] = 'vazio'
                dados[i]['vetor'] = None
                qtd_vazios[0] += 1
                return 
             # pilha de vetorização 
             _pos.append(i)
             _textos.append(dados[i].get('texto',''))
         UtilDocs.map_thread(__pre_avaliar__, range(len(dados)), n_threads=UtilDocs.N_THREADS_PADRAO, tdqm_progress=True)    
         #UtilDocs.progress_bar(1,1,' Concluído   
         bar.restantes(0)
         bar.close()
         self.printlog(f'Análise prévia: {qtd_vazios[0]} documentos estão vazios e {qtd_ok[0]} já possuem vetores                   ')
         self.printlog(f'Vetorizando {len(_textos)} textos ... ')
         vetores = self.vetores(_textos, normalizar=True, retornar_float=True)
         self.printlog('Consolidadno vetorização ...')
         for i, pos in enumerate(_pos):
               dados[pos]['vetor'] = vetores[i]
               dados[pos]['hash'] = UtilDocs.hash(_textos[i])

    def printlog(self,msg, retornar_msg_erro=False, destaque = False):
        UtilDocs.printlog('Doc2LLMRapido', msg, retornar_msg_erro, destaque)

    def tokens(self, textos):
        model = self.get_model()
        inputs = model.tokenizer(textos, padding='longest', truncation=True, max_length = self.max_length)
        return [e.tokens for e in inputs.encodings]   

    def preprocess_br(self, texto):
        return self.tokens([texto])[0]

    def teste_modelo(self):
         #print('Init Sims', end='')
         #self.model.init_sims(replace=True)
         #print(' _o/')
         texto_1 = 'esse é um texto de teste para comparação'
         texto_2 = 'temos aqui um outro texto de teste para uma nova comparação mais distante'
         texto_3 = 'esse é 1 texto para teste de comparação'
         texto_4 = 'esse é 1 texto para teste de comparação' * 200
         UtilDocs.print_linha()
         print(' >>>> TESTE DO MODELO <<<<')
         UtilDocs.print_linha('- ')
         print('- Texto 01:  ', texto_1)
         print('  > tokens: ', '|'.join(self.preprocess_br(texto_1)) )
         print('- Texto 02:  ', texto_2)
         print('  > tokens: ', '|'.join(self.preprocess_br(texto_2)) )
         print('- Texto 03:  ', texto_3)
         print('  > tokens: ', '|'.join(self.preprocess_br(texto_3)) )
         print('- Texto 04:  200 x Texto 03')
         UtilDocs.print_linha('- ')
         sim = 100*self.similaridade(texto_1, texto_1)
         print(f'Similaridade entre o texto 01 e ele mesmo: {sim:.2f}%')          
         sim = 100*self.similaridade(texto_1, texto_2)
         print(f'Similaridade entre os textos 01 e 02: {sim:.2f}%')          
         sim = 100*self.similaridade(texto_1, texto_3)
         print(f'Similaridade entre os textos 01 e 03: {sim:.2f}%')          
         sim = 100*self.similaridade(texto_3, texto_4)
         print(f'Similaridade entre os textos 03 e 04 (grande): {sim:.2f}%')          
         UtilDocs.print_linha()
         self.info()
         UtilDocs.print_linha()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Pasta do modelo')
    parser.add_argument('-pasta', help='pasta do modelo ou constante para modelo remoto - padrao T5BR', required=False)
    args = parser.parse_args()
    path = './tmpcache/modelos'

    PASTA_MODELO = str(args.pasta) or Doc2LLMRapido.T5BR

    dv = Doc2LLMRapido(modelo=PASTA_MODELO, path = path)
    dv.teste_modelo()

